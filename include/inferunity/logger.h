// 简单的日志系统框架
// 参考ONNX Runtime和TensorFlow的日志系统设计

#ifndef INFERUNITY_LOGGER_H
#define INFERUNITY_LOGGER_H

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <ctime>

namespace inferunity {

// 日志级别
enum class LogLevel {
    VERBOSE = 0,  // 详细调试信息
    INFO = 1,     // 一般信息
    WARNING = 2,  // 警告
    ERROR = 3,    // 错误
    FATAL = 4     // 致命错误
};

// 日志器类（单例模式）
class Logger {
public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }
    
    // 设置日志级别
    void SetLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }
    
    LogLevel GetLevel() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return level_;
    }
    
    // 设置是否输出到控制台
    void SetConsoleOutput(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        console_output_ = enable;
    }
    
    // 设置是否输出到文件
    void SetFileOutput(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_ = filename;
    }
    
    // 日志输出函数
    void Log(LogLevel level, const std::string& message, const std::string& file = "", int line = 0) {
        if (level < level_) {
            return;  // 低于当前级别，不输出
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string level_str = LogLevelToString(level);
        std::string timestamp = GetTimestamp();
        
        std::ostringstream oss;
        oss << "[" << timestamp << "] [" << level_str << "]";
        if (!file.empty()) {
            oss << " [" << file;
            if (line > 0) {
                oss << ":" << line;
            }
            oss << "]";
        }
        oss << " " << message << std::endl;
        
        std::string log_message = oss.str();
        
        // 输出到控制台
        if (console_output_) {
            if (level >= LogLevel::ERROR) {
                std::cerr << log_message;
            } else {
                std::cout << log_message;
            }
        }
        
        // 输出到文件（如果设置了）
        if (!log_file_.empty()) {
            // 简化实现：直接追加到文件
            // 实际应用中可以使用更复杂的文件管理
            std::ofstream ofs(log_file_, std::ios::app);
            if (ofs.is_open()) {
                ofs << log_message;
                ofs.close();
            }
        }
    }
    
private:
    Logger() : level_(LogLevel::INFO), console_output_(true) {}
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string LogLevelToString(LogLevel level) const {
        switch (level) {
            case LogLevel::VERBOSE: return "VERBOSE";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
    
    std::string GetTimestamp() const {
        // 简化实现：使用当前时间
        // 实际应用中可以使用更精确的时间格式化
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        // 使用线程安全的localtime_r（POSIX）或localtime_s（Windows）
        // 为了跨平台兼容，使用std::localtime（在线程安全锁内调用）
        std::tm tm_buf;
        #ifdef _WIN32
            localtime_s(&tm_buf, &time_t);
        #else
            localtime_r(&time_t, &tm_buf);
        #endif
        
        std::stringstream ss;
        ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    LogLevel level_;
    bool console_output_;
    std::string log_file_;
    mutable std::mutex mutex_;
};

// 便捷宏定义
#define LOG_VERBOSE(msg) inferunity::Logger::Instance().Log(inferunity::LogLevel::VERBOSE, msg, __FILE__, __LINE__)
#define LOG_INFO(msg) inferunity::Logger::Instance().Log(inferunity::LogLevel::INFO, msg, __FILE__, __LINE__)
#define LOG_WARNING(msg) inferunity::Logger::Instance().Log(inferunity::LogLevel::WARNING, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) inferunity::Logger::Instance().Log(inferunity::LogLevel::ERROR, msg, __FILE__, __LINE__)
#define LOG_FATAL(msg) inferunity::Logger::Instance().Log(inferunity::LogLevel::FATAL, msg, __FILE__, __LINE__)

} // namespace inferunity

#endif // INFERUNITY_LOGGER_H

