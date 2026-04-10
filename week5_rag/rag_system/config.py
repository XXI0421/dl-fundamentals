import os

# 模型配置（使用你验证过的 BGE，不再微调）
EMBEDDING_MODEL = 'BAAI/bge-base-zh'  # 或 bge-base-en-v1.5
RERANK_MODEL = 'BAAI/bge-reranker-base'

# HNSW 参数（来自 Day 3 最优配置）
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100  # 甜点配置：95.7%召回，0.3ms

# 分块参数（关键！影响检索质量）
CHUNK_SIZE = 150      # 每个块 token 数（约 300-400 汉字/英文词）
CHUNK_OVERLAP = 30   # 块间重叠，避免边界切断关键信息

# 检索配置
TOP_K_RETRIEVE = 50   # Bi-Encoder 召回数
TOP_K_RERANK = 10     # Cross-Encoder 精排后返回数

# Kimi API 配置
KIMI_API_KEY = os.getenv('KIMI_API_KEY', 'your_kimi_api_key_here')
KIMI_API_URL = 'https://api.moonshot.cn/v1/chat/completions'
KIMI_MODEL = 'moonshot-v1-8k'

# 启用 Kimi API（默认关闭，需要配置 API key 后开启）
ENABLE_KIMI_API = False
