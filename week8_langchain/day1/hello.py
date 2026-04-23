# mini_chain_demo.py - LangChain 迷你链演示示例
# 该文件演示了如何使用 LCEL (LangChain Expression Language) 创建一个简单的翻译链

# 从 langchain_core.prompts 模块导入 ChatPromptTemplate 类
# ChatPromptTemplate 用于创建聊天格式的提示词模板
from langchain_core.prompts import ChatPromptTemplate

# 从 langchain_core.output_parsers 模块导入 StrOutputParser 类
# StrOutputParser 用于将模型输出解析为字符串格式
from langchain_core.output_parsers import StrOutputParser

# 从 langchain_core.language_models.fake 模块导入 FakeListLLM 类
# FakeListLLM 是一个模拟的语言模型，用于测试，它会按顺序返回预设的响应
from langchain_core.language_models.fake import FakeListLLM

# ==================== 定义三个核心组件（积木）====================

# 创建提示词模板，模板内容为"将以下中文翻译成英文：{text}"
# {text} 是一个占位符，在调用时会被实际的输入文本替换
prompt = ChatPromptTemplate.from_template("将以下中文翻译成英文：{text}")

# 创建模拟的语言模型实例
# responses 参数指定了模型会按顺序返回的响应列表
# 这里设置为固定返回 "Hello world"，用于演示和测试
llm = FakeListLLM(responses=["Hello world"])  # 模拟 LLM，固定返回

# 创建输出解析器实例
# StrOutputParser 会将模型的输出（通常是一个 ChatMessage 对象）转换为纯字符串
parser = StrOutputParser()

# ==================== 使用 LCEL 管道操作符连接组件 ====================

# 使用 | 操作符（LCEL 管道操作符）将三个组件连接成一个链
# 数据流向：prompt(格式化输入) -> llm(生成响应) -> parser(解析输出)
# 这是 LangChain Expression Language 的核心特性
chain = prompt | llm | parser

# ==================== 运行链并获取结果 ====================
# 调用链的 invoke 方法，传入输入参数
# {"text": "你好世界"} 是输入字典，key "text" 对应模板中的 {text} 占位符
result = chain.invoke({"text": "你好世界"})

# 打印最终结果
# 由于 FakeListLLM 固定返回 "Hello world"，所以输出为 Hello world
print(result)  # 输出: Hello world