// 线程池实现
// 参考 ONNX Runtime 和 TensorFlow 的线程池设计

#include "inferunity/runtime.h"
#include "inferunity/logger.h"
#include <thread>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <vector>
#include <functional>

namespace inferunity {

// 线程池实现
class ThreadPoolImpl {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;  // 标记为mutable以便const方法使用
    std::condition_variable condition_;
    std::condition_variable finished_condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};  // 正在执行的任务数
    size_t thread_count_;
    
public:
    ThreadPoolImpl(size_t num_threads = 0) {
        if (num_threads == 0) {
            // 使用硬件并发数
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) {
                num_threads = 4;  // 默认4个线程
            }
        }
        thread_count_ = num_threads;
        
        // 创建工作线程
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                        active_tasks_++;
                    }
                    
                    // 执行任务
                    try {
                        task();
                    } catch (const std::exception& e) {
                        LOG_ERROR("Thread pool task exception: " + std::string(e.what()));
                    }
                    
                    // 任务完成，减少计数
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        active_tasks_--;
                        if (active_tasks_ == 0 && tasks_.empty()) {
                            finished_condition_.notify_all();
                        }
                    }
                }
            });
        }
        
        LOG_INFO("Thread pool created with " + std::to_string(num_threads) + " threads");
    }
    
    template<typename F>
    void Enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                LOG_WARNING("Thread pool is stopped, cannot enqueue task");
                return;
            }
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }
    
    void WaitAll() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        finished_condition_.wait(lock, [this] {
            return tasks_.empty() && active_tasks_ == 0;
        });
    }
    
    size_t GetThreadCount() const {
        return thread_count_;
    }
    
    size_t GetPendingTaskCount() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }
    
    ~ThreadPoolImpl() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        LOG_INFO("Thread pool destroyed");
    }
};

// 全局线程池实例
static std::unique_ptr<ThreadPoolImpl> g_thread_pool = nullptr;
static std::once_flag g_pool_init_flag;
static std::once_flag g_pool_destroy_flag;

ThreadPoolImpl* GetThreadPool(size_t num_threads = 0) {
    std::call_once(g_pool_init_flag, [num_threads]() {
        g_thread_pool = std::make_unique<ThreadPoolImpl>(num_threads);
        // 注册退出时清理函数
        std::atexit([]() {
            if (g_thread_pool) {
                g_thread_pool.reset();
            }
        });
    });
    return g_thread_pool.get();
}

// 公共接口实现
void ThreadPool::EnqueueTask(std::function<void()> task) {
    GetThreadPool()->Enqueue(std::move(task));
}

void ThreadPool::WaitAll() {
    GetThreadPool()->WaitAll();
}

size_t ThreadPool::GetThreadCount() {
    return GetThreadPool()->GetThreadCount();
}

size_t ThreadPool::GetPendingTaskCount() {
    return GetThreadPool()->GetPendingTaskCount();
}

} // namespace inferunity

