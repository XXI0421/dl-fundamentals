"""
Multi-Agent协作系统 - Streamlit前端
支持单Agent和多Agent两种模式
支持用户验收和回溯功能
"""
import streamlit as st
import os
import json
from typing import List, Dict, Any

# 设置页面配置
st.set_page_config(
    page_title="Multi-Agent协作系统",
    page_icon="🤖",
    layout="wide"
)

# 初始化会话状态
if 'mode' not in st.session_state:
    st.session_state.mode = 'single'  # 'single' or 'multi'
if 'requirement' not in st.session_state:
    st.session_state.requirement = ""
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'execution_result' not in st.session_state:
    st.session_state.execution_result = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_provider' not in st.session_state:
    st.session_state.api_provider = "kimi"
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'agent_outputs' not in st.session_state:
    st.session_state.agent_outputs = {}
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'show_accept_panel' not in st.session_state:
    st.session_state.show_accept_panel = False
if 'agent_config_confirmed' not in st.session_state:
    st.session_state.agent_config_confirmed = False
if 'additional_requirement' not in st.session_state:
    st.session_state.additional_requirement = ""

# 添加工具路径
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_llm_client():
    """加载LLM客户端"""
    from llm_client import KimiClient, OpenAIClient
    
    api_key = st.session_state.api_key
    provider = st.session_state.api_provider
    
    if not api_key:
        st.error("请先输入API密钥")
        return None
    
    try:
        if provider == "kimi":
            return KimiClient(api_key=api_key)
        else:
            return OpenAIClient(api_key=api_key)
    except Exception as e:
        st.error(f"初始化LLM客户端失败: {e}")
        return None

def run_single_agent(requirement: str):
    """运行单Agent模式"""
    from react_agent import ReActAgent
    from tools.base import ToolRegistry
    from tools.real_tools import init_default_tools
    from tools.python_sandbox import init_sandbox_tools
    
    llm_client = load_llm_client()
    if not llm_client:
        return None
    
    # 创建工具注册表
    registry = ToolRegistry()
    init_default_tools(registry)
    init_sandbox_tools(registry)
    
    # 创建Agent
    agent = ReActAgent(
        llm_client=llm_client,
        tool_registry=registry,
        max_iterations=8,
        memory_k=5
    )
    
    # 执行任务
    with st.spinner("Agent正在思考中..."):
        result = agent.run(requirement)
    
    return {
        "success": True,
        "response": result,
        "stats": agent.get_tool_call_stats()
    }

def analyze_and_generate_agents(requirement: str):
    """分析需求并生成Agent配置"""
    from multi_agent_coordinator import MultiAgentCoordinator
    
    llm_client = load_llm_client()
    if not llm_client:
        return None
    
    coordinator = MultiAgentCoordinator(llm_client=llm_client)
    
    with st.spinner("主Agent正在分析需求并生成Agent配置..."):
        result = coordinator.analyze_requirement_and_generate_agents(requirement)
    
    return result

def execute_agent_step(requirement: str, step_index: int):
    """执行单个Agent步骤"""
    from multi_agent_coordinator import MultiAgentCoordinator
    
    llm_client = load_llm_client()
    if not llm_client:
        return None
    
    # 从会话状态恢复coordinator
    coordinator = MultiAgentCoordinator(llm_client=llm_client)
    
    # 恢复Agent配置
    for agent in st.session_state.agents:
        coordinator.add_agent(agent['id'], agent['prompt'])
    
    # 设置当前步骤之前的输出
    for i in range(step_index):
        if f"output_{i}" in st.session_state.agent_outputs:
            coordinator.agents[i].output = st.session_state.agent_outputs[f"output_{i}"]
    
    with st.spinner(f"Agent {step_index + 1} 正在执行..."):
        result = coordinator.execute_agent_step(requirement, step_index)
    
    return result

# 主界面
st.title("🤖 Multi-Agent协作系统")

# API配置
with st.sidebar:
    st.header("API配置")
    
    # API提供商选择
    st.session_state.api_provider = st.selectbox(
        "LLM提供商",
        ["kimi", "openai"],
        index=0
    )
    
    # API密钥输入
    st.session_state.api_key = st.text_input(
        "API密钥",
        type="password",
        value=st.session_state.api_key,
        placeholder="请输入您的API密钥"
    )
    
    # 提示词模板示例
    st.subheader("提示词模板")
    st.code("""你是一个架构设计师，擅长设计系统架构。
请分析需求并输出完整的技术架构方案。""")

# 模式选择
st.radio(
    "选择运行模式",
    ["单Agent模式", "多Agent模式"],
    key="mode",
    horizontal=True
)

# 用户需求输入
st.text_area(
    "输入您的需求",
    key="requirement",
    height=100,
    placeholder="请描述您的需求，例如：设计一个待办事项应用的后端API..."
)

# 多Agent模式配置
if st.session_state.mode == "多Agent模式":
    st.subheader("配置子Agent")
    
    # 自动生成Agent按钮
    if st.session_state.requirement and not st.session_state.agent_config_confirmed:
        if st.button("🎯 由主Agent自动生成Agent配置"):
            result = analyze_and_generate_agents(st.session_state.requirement)
            if result:
                st.session_state.analysis_result = result
                st.session_state.agents = []
                for i, agent_config in enumerate(result.get("agents", [])):
                    st.session_state.agents.append({
                        "id": agent_config.get("id", f"Agent {i + 1}"),
                        "prompt": agent_config.get("prompt", ""),
                        "task": agent_config.get("task", ""),
                        "output": agent_config.get("output", f"output_{i}")
                    })
                st.success("主Agent已生成Agent配置，请确认后开始任务")
    
    # 显示已添加的Agent
    if st.session_state.agents:
        for i, agent in enumerate(st.session_state.agents):
            with st.expander(f"**Agent {i+1}: {agent.get('id', '未命名')}**"):
                agent['id'] = st.text_input(
                    f"Agent {i+1} 名称",
                    value=agent['id'],
                    key=f"agent_{i}_id"
                )
                agent['prompt'] = st.text_area(
                    f"Agent {i+1} 的提示词",
                    value=agent['prompt'],
                    key=f"agent_{i}_prompt",
                    height=100,
                    placeholder=f"例如：你是一个架构设计师，负责分析需求并设计系统架构。"
                )
                if 'task' in agent:
                    st.write(f"**任务**: {agent['task']}")
                if 'output' in agent:
                    st.write(f"**产出物**: {agent['output']}")
            
            # 删除按钮
            if st.button(f"删除 Agent {i+1}"):
                st.session_state.agents.pop(i)
                st.rerun()
        
        # 清空按钮
        if st.button("清空所有Agent"):
            st.session_state.agents = []
            st.session_state.agent_config_confirmed = False
            st.session_state.analysis_result = None
            st.rerun()
    
    # 手动添加Agent按钮
    if not st.session_state.agent_config_confirmed:
        if st.button("手动添加Agent"):
            st.session_state.agents.append({
                "id": f"Agent {len(st.session_state.agents) + 1}",
                "prompt": "",
                "task": "",
                "output": f"output_{len(st.session_state.agents)}"
            })
            st.rerun()

# 单Agent模式执行按钮
if st.session_state.mode == "单Agent模式":
    if st.session_state.requirement:
        if st.button("开始任务", disabled=st.session_state.is_running):
            st.session_state.is_running = True
            result = run_single_agent(st.session_state.requirement)
            st.session_state.execution_result = result
            st.session_state.is_running = False
            st.rerun()

# 多Agent模式执行流程
if st.session_state.mode == "多Agent模式":
    # 确认Agent配置
    if st.session_state.agents and not st.session_state.agent_config_confirmed:
        if st.button("确认Agent配置并开始任务"):
            st.session_state.agent_config_confirmed = True
            st.session_state.current_step = 0
            st.session_state.agent_outputs = {}
            st.session_state.show_accept_panel = False
            st.session_state.additional_requirement = ""
            st.rerun()
    
    # 执行步骤流程
    if st.session_state.agent_config_confirmed and st.session_state.agents:
        # 显示进度
        st.progress(st.session_state.current_step / len(st.session_state.agents))
        st.write(f"当前步骤: {st.session_state.current_step + 1} / {len(st.session_state.agents)}")
        
        # 显示当前步骤的Agent信息
        if st.session_state.current_step < len(st.session_state.agents):
            current_agent = st.session_state.agents[st.session_state.current_step]
            st.info(f"**当前执行Agent**: {current_agent['id']}")
            st.write(f"**任务**: {current_agent.get('task', '执行任务')}")
            
            # 执行按钮
            if f"output_{st.session_state.current_step}" not in st.session_state.agent_outputs:
                if st.button(f"执行 Agent {st.session_state.current_step + 1}"):
                    full_requirement = st.session_state.requirement
                    if st.session_state.additional_requirement:
                        full_requirement += "\n\n补充需求：" + st.session_state.additional_requirement
                    
                    result = execute_agent_step(full_requirement, st.session_state.current_step)
                    if result and result.get("success"):
                        st.session_state.agent_outputs[f"output_{st.session_state.current_step}"] = result["output"]
                        st.session_state.show_accept_panel = True
                        st.rerun()
            
            # 显示Agent输出和验收面板
            if f"output_{st.session_state.current_step}" in st.session_state.agent_outputs and st.session_state.show_accept_panel:
                st.subheader(f"Agent {st.session_state.current_step + 1} 产出")
                output = st.session_state.agent_outputs[f"output_{st.session_state.current_step}"]
                st.write(output)
                
                # 验收按钮
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ 验收通过，继续下一个Agent"):
                        st.session_state.current_step += 1
                        st.session_state.show_accept_panel = False
                        st.session_state.additional_requirement = ""
                        st.rerun()
                
                with col2:
                    if st.button("❌ 不通过，需要修改"):
                        st.session_state.show_accept_panel = False
            
            # 修改需求输入
            if not st.session_state.show_accept_panel and f"output_{st.session_state.current_step}" in st.session_state.agent_outputs:
                st.subheader("补充需求")
                st.session_state.additional_requirement = st.text_area(
                    "请输入需要补充的需求或修改意见",
                    value=st.session_state.additional_requirement,
                    height=100
                )
                if st.button("重新执行当前Agent"):
                    # 删除当前输出，重新执行
                    if f"output_{st.session_state.current_step}" in st.session_state.agent_outputs:
                        del st.session_state.agent_outputs[f"output_{st.session_state.current_step}"]
                    st.rerun()
        
        # 任务完成
        if st.session_state.current_step >= len(st.session_state.agents):
            st.success("🎉 所有Agent执行完成！")
            
            st.subheader("最终产出汇总")
            for key, value in st.session_state.agent_outputs.items():
                st.write(f"**{key}:**")
                st.write(value[:500] + "..." if len(str(value)) > 500 else value)
            
            # 重新开始按钮
            if st.button("🔄 开始新任务"):
                st.session_state.agent_config_confirmed = False
                st.session_state.current_step = 0
                st.session_state.agent_outputs = {}
                st.session_state.show_accept_panel = False
                st.session_state.additional_requirement = ""
                st.session_state.analysis_result = None
                st.rerun()

# 显示单Agent模式执行结果
if st.session_state.mode == "单Agent模式" and st.session_state.execution_result:
    result = st.session_state.execution_result
    
    if result.get("success"):
        st.success("任务执行完成！")
        
        st.subheader("Agent响应")
        st.write(result["response"])
        
        if result.get("stats"):
            st.subheader("执行统计")
            stats = result["stats"]
            st.write(f"步骤数: {stats['total_steps']}")
            st.write(f"成功: {stats['successful_steps']}")
            st.write(f"失败: {stats['failed_steps']}")
            if stats['tools_used']:
                st.write(f"使用工具: {', '.join(stats['tools_used'])}")
    else:
        st.error(f"任务执行失败: {result.get('error', '未知错误')}")

# 页脚
st.markdown("---")
st.markdown("Powered by Streamlit & Multi-Agent System")
