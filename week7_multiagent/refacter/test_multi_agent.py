"""
多Agent协调器测试
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_coordinator import MultiAgentCoordinator
from llm_client import get_llm_client
from config import Config

def test_multi_agent_coordinator():
    """测试多Agent协调器"""
    print("测试多Agent协调器...")
    
    # 检查API密钥
    if not Config.is_llm_configured():
        print("⚠️ 未配置LLM API密钥，跳过测试")
        return
    
    try:
        # 创建LLM客户端
        llm_client = get_llm_client()
        
        # 创建协调器
        coordinator = MultiAgentCoordinator(llm_client=llm_client)
        
        # 添加测试Agent
        coordinator.add_agent("分析员", "你是一个需求分析专家，负责分析用户需求。")
        coordinator.add_agent("设计师", "你是一个系统设计师，负责设计技术方案。")
        
        print("✅ 协调器创建成功，已添加Agent")
        
        # 测试需求分析
        requirement = "设计一个简单的待办事项应用"
        analysis = coordinator.analyze_requirement_and_generate_agents(requirement)
        print(f"✅ 需求分析结果: {analysis['task_name']}")
        print(f"   生成Agent数量: {len(analysis.get('agents', []))}")
        
        # 测试执行单个步骤
        print("\n测试单步执行...")
        step_result = coordinator.execute_agent_step(requirement, 0)
        print(f"✅ 单步执行结果: {step_result.get('success', False)}")
        
        if step_result.get('success'):
            print(f"   Agent: {step_result.get('agent_name')}")
            print(f"   产出: {step_result.get('output', '')[:50]}...")
        
        print("\n🎉 多Agent协调器测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_agent_coordinator()
