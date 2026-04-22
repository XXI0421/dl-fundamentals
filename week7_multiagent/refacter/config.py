"""
配置文件 - 管理API密钥和运行参数
"""
import os
from typing import Optional

class Config:
    # LLM配置
    KIMI_API_KEY: str = os.environ.get("KIMI_API_KEY", "")
    KIMI_MODEL: str = os.environ.get("KIMI_MODEL", "moonshot-v1-128k")
    
    # OpenAI兼容配置（可选）
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    
    # 记忆系统配置
    LONG_TERM_MEMORY_PATH: str = "./memory_store"
    EMBEDDING_MODEL: str = "BAAI/bge-base-zh"
    
    # 执行配置
    MAX_ITERATIONS: int = 8
    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 1.0
    
    # 输出配置
    OUTPUT_DIR: str = "./output"
    
    @classmethod
    def is_llm_configured(cls) -> bool:
        """检查是否配置了至少一个LLM API"""
        return bool(cls.KIMI_API_KEY) or bool(cls.OPENAI_API_KEY)
    
    @classmethod
    def get_default_llm_provider(cls) -> Optional[str]:
        """获取默认LLM提供商"""
        if cls.KIMI_API_KEY:
            return "kimi"
        elif cls.OPENAI_API_KEY:
            return "openai"
        return None
