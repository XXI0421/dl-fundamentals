"""
多工具协作演示程序
演示场景：分析 2025年 Python 在 GitHub 上的趋势，并生成可视化报告

工具协作流程：
1. 使用 python_sandbox（网络模式）搜索数据
2. 使用 python_sandbox 执行 Python 代码进行数据分析和绘图
3. 使用 save_file 保存分析报告和图表

特性演示：
- 工具调用重试机制（指数退避）
- 无效JSON工具调用自动修正
- 多工具协作完成复杂任务
- 支持网络请求获取实时数据
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v4 import MultiToolAgent
from tools.base import ToolRegistry
from tools.real_tools import (
    search_duckduckgo, 
    save_file,
    read_file,
    get_current_year
)
from tools.python_sandbox import python_sandbox, calculator


def print_banner():
    """显示演示程序横幅"""
    print("=" * 70)
    print("      ReAct Agent V4 - 多工具协作演示")
    print("=" * 70)
    print("任务：分析 2025年 Python 在 GitHub 上的趋势，并生成可视化报告")
    print("工具链：搜索数据 → Python分析 → 生成图表 → 保存报告")
    print("模式：网络请求已启用")
    print("=" * 70)


def main():
    # 检查API密钥
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY")
        api_key = input("请输入API密钥: ").strip()
        if not api_key:
            print("错误：API密钥不能为空")
            return
        os.environ["KIMI_API_KEY"] = api_key
    
    # 初始化工具注册表
    registry = ToolRegistry()
    
    # 注册所有需要的工具
    registry.register(search_duckduckgo)
    registry.register(python_sandbox)
    registry.register(save_file)
    registry.register(read_file)
    registry.register(get_current_year)
    registry.register(calculator)
    
    print(f"📦 已注册工具: {', '.join(registry.list_tools())}")
    
    # 初始化Agent（配置重试机制）
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"), model="moonshot-v1-128k")
    agent = MultiToolAgent(
        client, 
        registry, 
        max_iterations=8,
        memory_k=5, 
        ltm_threshold=0.3, 
        ltm_max_facts=3,
        max_retries=3,
        retry_base_delay=1.0,
        retry_multiplier=2.0
    )
    
    print_banner()
    print(f"\n🚀 已创建新会话，会话ID: {agent.get_session_id()}")
    
    # 定义复杂任务（允许网络模式）
    complex_task = """
分析 2025年 Python 在 GitHub 上的趋势，并生成可视化报告。

请按以下步骤完成：
1. 使用 python_sandbox 工具，设置 allow_network=True，通过网络请求获取 GitHub 上 Python 相关的10个热门仓库数据
2. 分析获取的数据（star数量、fork数量、更新时间等）
3. 使用 python_sandbox 执行 Python 代码进行数据分析和可视化图表生成
4. (最重要)可视化图表生成的脚本要求：x轴项目名称进行旋转、避免重叠，同时显示y轴数据（禁用科学计数）
5. 默认不启用长期记忆

可用工具说明：
- python_sandbox: 执行Python代码，支持参数 mode='full' 或 allow_network=True 启用网络请求
- save_file: 保存内容到文件

请提供详细的分析结果和可视化图表。
"""
    
    print("\n🤔 正在处理复杂任务...")
    print("任务内容:", complex_task.strip()[:100], "...")
    
    try:
        # 执行任务
        response = agent.run(complex_task)
        
        print("\n" + "=" * 70)
        print("📊 任务执行结果")
        print("=" * 70)
        print(f"\n🤖 Agent 最终回答:\n{response}")
        
        # 打印工具调用统计
        stats = agent.get_tool_call_stats()
        print("\n📈 工具调用统计:")
        print(f"  总步骤数: {stats['total_steps']}")
        print(f"  成功步骤: {stats['successful_steps']}")
        print(f"  失败步骤: {stats['failed_steps']}")
        print(f"  总重试次数: {stats.get('total_retries', 0)}")
        print(f"  使用工具: {', '.join(stats['tools_used'])}")
        
        print("\n🚶 执行轨迹:")
        for step in stats['trajectory']:
            status = "✅" if step['success'] else "❌"
            retry_info = f"(重试{step['retry_count']}次)" if step['retry_count'] > 0 else ""
            print(f"  {status} 步骤{step['step']}: {step['tool']} {retry_info} - {step['observation']}")
        
    except Exception as e:
        print(f"\n❌ 任务执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()