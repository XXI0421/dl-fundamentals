"""
工具模块测试
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.base import ToolRegistry
from tools.real_tools import init_default_tools
from tools.python_sandbox import init_sandbox_tools

def test_tools():
    """测试工具模块"""
    print("测试工具模块...")
    
    # 创建工具注册表
    registry = ToolRegistry()
    
    # 初始化默认工具
    init_default_tools(registry)
    init_sandbox_tools(registry)
    
    print(f"✅ 注册工具: {registry.list_tools()}")
    
    # 测试get_current_year
    result = registry.execute("get_current_year", {})
    print(f"✅ 测试get_current_year: {result}")
    
    # 测试python_calculator
    result = registry.execute("python_calculator", {"expression": "2 + 3 * 4"})
    print(f"✅ 测试python_calculator: {result}")
    
    # 测试python_sandbox
    result = registry.execute("python_sandbox", {"code": "print('Hello, World!')"})
    print(f"✅ 测试python_sandbox: {result[:50]}...")
    
    # 测试save_file和read_file
    test_content = "测试内容"
    save_result = registry.execute("save_file", {"filename": "test_output.txt", "content": test_content})
    print(f"✅ 测试save_file: {save_result}")
    
    # 检查文件是否保存到output目录
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "test_output.txt")
    read_result = registry.execute("read_file", {"filename": "test_output.txt"})
    print(f"✅ 测试read_file: {read_result}")
    
    # 清理测试文件
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith("test_"):
                os.remove(os.path.join(output_dir, file))
        # 如果目录为空，删除目录
        if not os.listdir(output_dir):
            os.rmdir(output_dir)
    
    print("\n🎉 所有工具测试通过！")

if __name__ == "__main__":
    test_tools()
