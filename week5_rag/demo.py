"""
BGE模型交互探索工具
用于测试和展示BGE模型的语义检索能力
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from agent_data import ALL_CONCEPTS


class BGEExplorer:
    """BGE模型探索类"""
    
    def __init__(self, model_name='BAAI/bge-base-zh'):
        """
        初始化BGE模型探索器
        
        Args:
            model_name: 模型名称
        """
        print("[INFO] 加载BGE模型...")
        print(f"   模型: {model_name}")
        # 加载BGE模型
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # 构建文档库
        self.build_corpus()
        print(f"[OK] 就绪！共 {len(self.corpus)} 个文档")
        print("="*60)
    
    def build_corpus(self):
        """构建Agent领域文档库"""
        self.corpus = []
        self.metadata = []
        
        # 遍历所有概念数据
        for concept_data in ALL_CONCEPTS:
            concept = concept_data["concept"]
            
            # 添加正例文档
            for doc in concept_data["positives"]:
                self.corpus.append(doc)
                self.metadata.append({
                    "concept": concept,
                    "type": "正例",
                    "category": "Agent技术"
                })
            
            # 添加干扰项文档
            for doc in concept_data["hard_negatives"]:
                self.corpus.append(doc)
                self.metadata.append({
                    "concept": concept,
                    "type": "干扰项",
                    "category": "其他领域"
                })
        
        print("   编码文档库...")
        # 生成文档嵌入
        self.doc_embeddings = self.model.encode(
            self.corpus, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def search(self, query, top_k=5):
        """执行语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            list: 检索结果列表
        """
        # 生成查询嵌入
        query_emb = self.model.encode(query, convert_to_numpy=True)
        
        # 计算相似度
        similarities = cosine_similarity([query_emb], self.doc_embeddings)[0]
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "score": float(similarities[idx]),
                "text": self.corpus[idx],
                "concept": self.metadata[idx]["concept"],
                "type": self.metadata[idx]["type"]
            })
        
        return results
    
    def display_results(self, query, results):
        """美观的结果展示
        
        Args:
            query: 查询文本
            results: 检索结果
        """
        print(f"\n[QUERY] '{query}'")
        print("-" * 60)
        print(f"{'Rank':<6} {'Score':<8} {'Type':<8} {'Concept':<15} Text")
        print("-" * 60)
        
        for r in results:
            # 为正例和干扰项添加标记
            marker = "[+]" if r['type'] == "正例" else "[-]"
            # 截断长文本
            text_short = r['text'][:40] + "..." if len(r['text']) > 40 else r['text']
            print(f"{r['rank']:<6} {r['score']:.3f}    {marker} {r['type']:<6} {r['concept']:<12} {text_short}")
        
        # 统计正例数量
        pos_count = sum(1 for r in results if r['type'] == "正例")
        print("-" * 60)
        print(f"[STATS] Top-{len(results)}中: 正例{pos_count}个, 干扰项{len(results)-pos_count}个")
        
        # 根据正例数量给出评价
        if pos_count == 0:
            print("[WARN] 前5个结果全是干扰项")
        elif pos_count >= 3:
            print("[OK] BGE成功识别了相关概念")
    
    def analyze_query(self, query):
        """深度分析查询语义
        
        Args:
            query: 查询文本
        """
        print(f"\n[ANALYSIS] 语义分析: '{query}'")
        
        # 生成查询嵌入
        emb = self.model.encode(query, convert_to_numpy=True)
        print(f"   向量维度: {emb.shape}")
        print(f"   向量范数: {np.linalg.norm(emb):.3f}")
        
        # 执行检索并展示结果
        results = self.search(query, top_k=5)
        self.display_results(query, results)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*60)
        print("[INTERACTIVE MODE]")
        print("命令:")
        print("  - 直接输入查询文本")
        print("  - 'test': 运行标准测试集")
        print("  - 'concept <概念名>': 查看某概念的所有文档")
        print("  - 'quit': 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n[INPUT] 请输入: ").strip()
                
                if not user_input:
                    continue
                
                # 退出命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("[BYE]")
                    break
                
                # 运行测试集
                if user_input.lower() == 'test':
                    self.run_standard_tests()
                    continue
                
                # 查看概念文档
                if user_input.lower().startswith('concept '):
                    concept_name = user_input[8:].strip()
                    self.show_concept_docs(concept_name)
                    continue
                
                # 分析查询
                self.analyze_query(user_input)
                
            except KeyboardInterrupt:
                print("\n[BYE]")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
    
    def run_standard_tests(self):
        """运行标准测试集 - 检查是否命中正确概念"""
        # 测试用例
        test_cases = [
            ("ReAct是什么", "ReAct"),
            ("Chain-of-Thought原理", "Chain-of-Thought"),
            ("Tool Use能力", "Tool Use"),
            ("RAG如何解决幻觉", "RAG"),
            ("Agent Planning方法", "Planning"),
            ("多智能体协作", "Multi-Agent"),
            ("System Prompt作用", "System Prompt"),
            ("零样本学习", "Zero-shot"),
            ("什么是注意力机制", "Attention Mechanism"),
            ("解释反向传播", "Backpropagation"),
        ]
        
        print(f"\n[TEST] 运行标准测试 ({len(test_cases)}个查询)...")
        print("-" * 60)
        
        top1_correct = 0
        top3_correct = 0
        
        # 遍历测试用例
        for query, expected_concept in test_cases:
            results = self.search(query, top_k=3)
            
            # 获取Top-1和Top-3概念
            top1_concept = results[0]['concept']
            top3_concepts = [r['concept'] for r in results]
            
            # 检查是否命中正确概念
            is_top1_correct = expected_concept.lower() in top1_concept.lower()
            is_top3_correct = any(expected_concept.lower() in c.lower() for c in top3_concepts)
            
            if is_top1_correct:
                top1_correct += 1
            if is_top3_correct:
                top3_correct += 1
            
            # 打印测试结果
            status = "[OK]" if is_top1_correct else "[X]"
            print(f"{status} {query[:20]:<20} | 期望: {expected_concept:<15} | Top-1: {top1_concept}")
        
        # 打印测试统计
        print("-" * 60)
        print(f"[RESULT] Top-1 准确率: {top1_correct}/{len(test_cases)} = {top1_correct/len(test_cases)*100:.1f}%")
        print(f"[RESULT] Top-3 命中率: {top3_correct}/{len(test_cases)} = {top3_correct/len(test_cases)*100:.1f}%")
        print(f"\n[NOTE] Top-1 准确率才是真正的检索质量指标")
    
    def show_concept_docs(self, concept_name):
        """展示某个概念的所有文档
        
        Args:
            concept_name: 概念名称
        """
        found = False
        print(f"\n[CONCEPT] '{concept_name}' 的文档:")
        
        # 遍历所有概念
        for c in ALL_CONCEPTS:
            if c["concept"].lower() == concept_name.lower():
                found = True
                # 打印正例
                print(f"\n正例 ({len(c['positives'])}个):")
                for i, doc in enumerate(c['positives'], 1):
                    print(f"  {i}. {doc}")
                
                # 打印干扰项
                print(f"\n干扰项 ({len(c['hard_negatives'])}个):")
                for i, doc in enumerate(c['hard_negatives'], 1):
                    print(f"  {i}. {doc}")
                break
        
        # 概念未找到
        if not found:
            print(f"[WARN] 未找到概念 '{concept_name}'")
            print(f"可用概念: {', '.join([c['concept'] for c in ALL_CONCEPTS[:5]])}...")


if __name__ == "__main__":
    # 初始化探索器
    explorer = BGEExplorer()
    # 进入交互模式
    explorer.interactive_mode()
