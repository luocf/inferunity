#!/bin/bash
# 准备提交到GitHub的脚本

set -e

cd "$(dirname "$0")/.."

echo "=== 准备提交到GitHub ==="
echo ""

# 1. 初始化git仓库（如果还没有）
if [ ! -d .git ]; then
    echo "1. 初始化Git仓库..."
    git init
    echo "   ✓ Git仓库已初始化"
else
    echo "1. Git仓库已存在"
fi

# 2. 检查是否有未提交的更改
echo ""
echo "2. 检查文件状态..."
if [ -d .git ]; then
    git status --short | head -20
    echo ""
    echo "   总文件数: $(git ls-files | wc -l | tr -d ' ')"
fi

# 3. 显示下一步操作
echo ""
echo "=== 下一步操作 ==="
echo ""
echo "1. 添加所有文件到暂存区："
echo "   git add ."
echo ""
echo "2. 提交更改："
echo "   git commit -F COMMIT_MESSAGE.txt"
echo "   或者："
echo "   git commit -m \"feat: 完成推理引擎核心功能实现\""
echo ""
echo "3. 在GitHub上创建新仓库（如果还没有）："
echo "   - 访问 https://github.com/new"
echo "   - 仓库名: ifusionengine"
echo "   - 选择 Public 或 Private"
echo "   - 不要初始化README、.gitignore或license（我们已经有了）"
echo ""
echo "4. 添加远程仓库并推送："
echo "   git remote add origin https://github.com/你的用户名/ifusionengine.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=== 当前项目统计 ==="
echo "  源代码文件: $(find src include -name '*.cpp' -o -name '*.h' 2>/dev/null | wc -l | tr -d ' ')"
echo "  文档文件: $(find docs -name '*.md' 2>/dev/null | wc -l | tr -d ' ')"
echo "  测试文件: $(find tests -name '*.cpp' 2>/dev/null | wc -l | tr -d ' ')"
echo ""

