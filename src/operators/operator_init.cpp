// 算子初始化函数
// 通过强制链接所有算子源文件来确保静态初始化被执行

#include "inferunity/operator.h"
#include <iostream>

namespace inferunity {

void InitializeOperators() {
    // 方法1: 通过访问OperatorRegistry单例来触发初始化
    // 但这不能保证静态变量被初始化
    
    // 方法2: 使用函数指针数组来强制引用所有静态变量
    // 由于REGISTER_OPERATOR宏创建了静态变量，我们需要确保它们被引用
    
    // 实际上，问题在于：静态库中的未使用符号可能被链接器优化掉
    // 解决方案：在CMakeLists.txt中使用-Wl,--whole-archive或确保所有符号被链接
    
    // 临时解决方案：直接访问注册表，这会触发单例初始化
    // 但静态变量的初始化仍然依赖于链接器是否包含它们
    auto& registry = OperatorRegistry::Instance();
    (void)registry;
    
    // 更好的方法：在CMakeLists.txt中确保所有算子源文件都被链接
    // 或者使用-Wl,--whole-archive来强制包含所有符号
}

} // namespace inferunity
