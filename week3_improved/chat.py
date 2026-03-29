import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import threading

st.set_page_config(page_title="Qwen2.5-7B 数学助手", page_icon="🧮", layout="wide")

# 注入 KaTeX CSS 优化数学公式显示
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<style>
    .math-inline { display: inline-block; margin: 0 0.2em; }
    .math-block { display: block; margin: 1em 0; text-align: center; overflow-x: auto; }
    div[data-testid="stMarkdownContainer"] p { line-height: 1.8; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", 
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    return model, tokenizer

# ==================== 侧边栏参数 + 数学模式设置 ====================
with st.sidebar:
    st.header("⚙️ 生成参数")
    
    temperature = st.slider("Temperature", 0.1, 1.5, 0.3, 0.1, 
                           help="数学推导建议 0.2-0.4（精确），创意任务可 0.7+")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    max_new_tokens = st.slider("Max Tokens", 128, 4096, 2048, 128,
                              help="数学证明建议 2048+")
    repetition_penalty = st.slider("重复惩罚", 1.0, 2.0, 1.05, 0.05)
    
    st.divider()
    st.header("🧮 数学输出设置")
    
    # 关键：强制模型使用 LaTeX 输出数学符号
    force_latex = st.toggle("强制 LaTeX 数学格式", value=True, 
                           help="要求模型必须用 $...$ 或 $$...$$ 输出所有数学符号")
    
    render_engine = st.selectbox("公式渲染引擎", ["默认 Markdown", "MathJax 优化"], index=0)
    
    st.divider()
    
    if st.button("🗑️ 清除对话", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ==================== 系统提示词（关键改进）====================
# 根据是否开启强制 LaTeX 模式，动态调整系统提示词
if force_latex:
    default_system_prompt = """你是一个数学助手。请严格遵守以下输出规则：

1. **所有数学符号必须用 LaTeX 格式**：
   - 行内公式使用单个美元符号：$E=mc^2$, $\\mathbf{F}=(F_x,F_y,F_z)$
   - 独立公式使用双美元符号：$$\\nabla \\times \\mathbf{F} = \\left( \\frac{\\partial F_z}{\\partial y} - \\frac{\\partial F_y}{\\partial z}, ... \\right)$$
   
2. **希腊字母必须使用 LaTeX**：$\\alpha$, $\\beta$, $\\gamma$, $\\theta$, $\\lambda$ 等，禁止直接写 "alpha"

3. **向量/矩阵使用粗体**：$\\mathbf{x}$, $\\mathbf{A}$, $\\nabla$（倒三角），$\\partial$（偏导）

4. **运算符规范**：$\\times$（叉乘），$\\cdot$（点乘），$\\int$, $\\sum$, $\\prod$

5. **分数和上下标**：$\\frac{a}{b}$, $x^{i}$, $y_{j}$

记住：用户看到的应该是排版精美的数学符号，而不是 LaTeX 源代码。"""
else:
    default_system_prompt = "你是一个 helpful 的 AI 助手。"

system_prompt = st.sidebar.text_area("系统提示词", value=default_system_prompt, height=250)

# ==================== 主界面 ====================
with st.spinner("加载模型中..."):
    model, tokenizer = load_model()

st.title("🧮 Qwen2.5-7B 本地模型")
st.caption("支持 LaTeX 公式渲染 | 自动数学符号格式化" + (" | ✅ LaTeX 强制模式开启" if force_latex else ""))

if "history" not in st.session_state:
    st.session_state.history = []

# 构造消息（带系统提示词）
def build_messages():
    return [{"role": "system", "content": system_prompt}] + st.session_state.history

# 显示历史对话（关键：使用 help参数确保 LaTeX 渲染）
for h in st.session_state.history:
    with st.chat_message(h["role"]):
        # 使用 unsafe_allow_html=True 确保复杂 LaTeX 能渲染
        st.markdown(h["content"], unsafe_allow_html=True)

# 输入区域
prompt = st.chat_input("输入问题")
if prompt:
    # 如果用户输入也包含 LaTeX，先显示
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)
    
    # 准备生成
    messages = build_messages()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            use_cache=True
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 流式显示（累积 LaTeX）
        for text_chunk in streamer:
            full_response += text_chunk
            # 实时渲染，添加光标效果
            placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        
        # 最终渲染（去掉光标）
        placeholder.markdown(full_response, unsafe_allow_html=True)
    
    st.session_state.history.append({"role": "assistant", "content": full_response})
