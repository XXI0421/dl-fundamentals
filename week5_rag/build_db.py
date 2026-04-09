"""
脚本1：构建Agent领域专用向量数据库
从agent_data.py加载结构化概念，构建HNSW索引并保存
"""

import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# 导入所有领域数据（假设agent_data.py在同目录）
# 数据结构：ALL_CONCEPTS = [{concept, description, positives, hard_negatives}, ...]
try:
    from agent_data import ALL_CONCEPTS
except ImportError:
    print("Error: agent_data.py not found. Please ensure it's in the same directory.")
    exit(1)


def build_agent_corpus():
    """
    构建语料库：每个概念的positives作为独立文档，保留concept标签用于评估
    """
    corpus = []          # 文档文本列表
    metadata = []        # 元数据（用于追踪属于哪个概念）
    
    print(f"Loading {len(ALL_CONCEPTS)} concepts from all domains...")
    
    for item in ALL_CONCEPTS:
        concept_name = item["concept"]
        
        # 将每个positive描述作为独立文档
        for idx, desc in enumerate(item["positives"]):
            corpus.append(desc)
            metadata.append({
                "concept": concept_name,
                "type": "positive",
                "idx": idx,
                "description": item["description"]  # 保留简短定义
            })
        
        # 可选：也将description本身加入语料（作为标准定义）
        corpus.append(f"{concept_name}: {item['description']}")
        metadata.append({
            "concept": concept_name,
            "type": "definition",
            "idx": -1,
            "description": item["description"]
        })
    
    print(f"Built corpus with {len(corpus)} documents from {len(ALL_CONCEPTS)} concepts")
    return corpus, metadata


def encode_and_index(corpus, model_name='BAAI/bge-base-zh'):
    """
    编码语料库并构建FAISS HNSW索引
    """
    print(f"\nInitializing model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 动态获取维度（应为768）
    test_emb = model.encode("test", convert_to_numpy=True)
    dim = test_emb.shape[0]
    print(f"Model dimension: {dim}")
    
    print(f"\nEncoding {len(corpus)} documents...")
    # 批量编码（32或64取决于内存）
    embeddings = model.encode(
        corpus, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        batch_size=32
    ).astype('float32')
    
    # L2归一化（使用内积=余弦相似度）
    faiss.normalize_L2(embeddings)
    
    # 构建HNSW索引（使用Day 3最优参数）
    print("\nBuilding HNSW index...")
    index = faiss.IndexHNSWFlat(dim, 16)  # M=16
    index.hnsw.efConstruction = 200      # 构建期搜索宽度
    index.add(embeddings)                  # 添加向量
    index.hnsw.efSearch = 100              # 默认查询参数
    
    print(f"✅ Index built: {index.ntotal} vectors, {dim} dimensions")
    
    return index, embeddings, model


def save_database(corpus, metadata, index, embeddings, output_dir="agent_db"):
    """
    保存完整数据库到本地
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存语料和元数据（JSON格式）
    with open(f"{output_dir}/corpus.jsonl", "w", encoding="utf-8") as f:
        for doc, meta in zip(corpus, metadata):
            line = {
                "text": doc,
                "meta": meta
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    # 2. 保存FAISS索引（二进制）
    faiss.write_index(index, f"{output_dir}/agent.index")
    
    # 3. 保存向量（numpy）
    np.save(f"{output_dir}/embeddings.npy", embeddings)
    
    # 4. 保存元数据（pickle，保留Python对象结构）
    with open(f"{output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ Database saved to ./{output_dir}/")
    print(f"   - corpus.jsonl: {len(corpus)} documents")
    print(f"   - agent.index: FAISS HNSW index")
    print(f"   - embeddings.npy: vector matrix")
    print(f"   - metadata.pkl: document metadata")


if __name__ == "__main__":
    print("="*60)
    print("Agent Domain Vector Database Builder")
    print("="*60)
    
    # 1. 构建语料
    corpus, metadata = build_agent_corpus()
    
    # 2. 编码并建索引
    index, embeddings, model = encode_and_index(corpus)
    
    # 3. 保存到本地
    save_database(corpus, metadata, index, embeddings)
    
    # 4. 打印统计
    print("\n" + "="*60)
    print("Database Statistics")
    print("="*60)
    print(f"Total concepts: {len(ALL_CONCEPTS)}")
    print(f"Total documents: {len(corpus)}")
    print(f"Average doc length: {sum(len(d) for d in corpus)/len(corpus):.0f} chars")
    print(f"Concepts included: {[item['concept'] for item in ALL_CONCEPTS[:5]]} ...")
