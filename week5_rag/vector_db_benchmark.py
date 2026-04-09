import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class HNSWBenchmark:
    """HNSW索引性能基准测试类（修复版）"""
    
    def __init__(self, corpus_path="corpus.jsonl", queries_path="queries.jsonl"):
        """
        初始化基准测试类
        
        Args:
            corpus_path: 语料库文件路径（文档）
            queries_path: 查询文件路径（独立用户问题）
        """
        # 初始化SentenceTransformer模型，用于将文本编码为向量
        # 'BAAI/bge-base-zh'是中文嵌入模型，也可替换为英文模型如'bge-base-en-v1.5'
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')  # 或换英文模型如'bge-base-en-v1.5'
        # 设置向量维度为768（bge-base-zh模型的输出维度）
        self.dim = 768
        
        # 加载语料库（文档），调用load_jsonl方法解析JSONL文件
        self.corpus = self.load_jsonl(corpus_path)
        # 加载独立查询集（真实用户问题，非语料库样本），同样调用load_jsonl
        self.queries = self.load_jsonl(queries_path)
        
        # 初始化FAISS索引对象，初始状态为None，待build_index方法构建
        self.index = None
        # 初始化语料库向量存储，用于后续的暴力搜索计算真值
        self.corpus_embeddings = None
        
        # 输出加载的数据统计信息：语料库文档数和查询集问题数
        print(f"Corpus: {len(self.corpus)} docs | Queries: {len(self.queries)} questions")
    
    def load_jsonl(self, path):
        """加载JSONL文件"""
        # 检查文件路径是否存在，避免文件不存在导致的错误
        if not os.path.exists(path):
            # 若文件不存在，打印警告信息并返回空列表
            print(f"Warning: {path} not found")
            return []
        
        # 初始化空列表，用于存储从JSONL文件中提取的文本内容
        items = []
        # 以只读模式打开JSONL文件，使用UTF-8编码支持多语言字符
        with open(path, "r", encoding="utf-8") as f:
            # 遍历文件的每一行，每行对应一个JSON对象
            for line in f:
                # 尝试解析JSON并提取"text"字段的值
                try:
                    # 解析当前行的JSON字符串，转换为Python字典
                    # 提取字典中键为"text"的值（即文档或查询内容），添加到列表
                    items.append(json.loads(line)["text"])
                except:
                    # 若解析失败（格式错误或缺少"text"字段），跳过该行继续处理
                    continue
        # 返回包含所有文本项的列表
        return items
    
    def build_index(self, m=16, ef_construction=200):
        """构建HNSW索引（与原版相同）"""
        # 检查语料库是否为空，空则无法构建索引，输出错误并返回
        if not self.corpus:
            print("Error: Empty corpus")
            return
        
        # 输出索引构建参数信息：M（每个节点的最大连接数）和ef_construction（构建时的搜索深度）
        print(f"\nBuilding index (M={m}, ef_construction={ef_construction})...")
        
        # 设置批处理大小为1000，避免一次性编码过多文本导致内存溢出
        batch_size = 1000
        # 初始化空列表，用于存储分批编码后的向量数组
        all_embeddings = []
        
        # 使用tqdm创建进度条，遍历语料库按batch_size分批处理
        # range(0, len(self.corpus), batch_size)生成批次起始索引序列
        for i in tqdm(range(0, len(self.corpus), batch_size), desc="Encoding corpus"):
            # 从语料库中提取当前批次的文本切片（i到i+batch_size）
            batch = self.corpus[i:i+batch_size]
            # 使用SentenceTransformer模型对当前批次文本进行向量化编码
            # show_progress_bar=False禁用内部进度条，避免与tqdm重复显示
            # convert_to_numpy=True确保输出为numpy数组格式
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            # 将编码结果转换为float32类型（FAISS要求），并添加到列表
            all_embeddings.append(emb.astype('float32'))
        
        # 使用np.vstack将列表中的所有向量数组垂直堆叠成一个完整的二维数组
        # 结果形状为(num_docs, dim)，即(文档数, 768)
        embeddings = np.vstack(all_embeddings)
        # 对向量进行L2归一化（转换为单位向量），使点积等价于余弦相似度
        faiss.normalize_L2(embeddings)
        
        # 创建HNSW索引实例，使用Flat存储基础向量（ brute-force搜索后备）
        # m参数控制图的连通性（每个节点在图中保持的邻居数），影响召回率和构建时间
        self.index = faiss.IndexHNSWFlat(self.dim, m)
        # 设置索引构建阶段的搜索深度，越大图质量越高但构建越慢
        self.index.hnsw.efConstruction = ef_construction
        # 将所有归一化后的文档向量添加到HNSW索引中，构建图结构
        self.index.add(embeddings)
        # 保存文档向量到实例变量，供后续的暴力搜索计算真值使用
        self.corpus_embeddings = embeddings
        
        # 输出索引构建完成信息，显示索引中实际存储的文档总数
        print(f"✅ Indexed {self.index.ntotal} documents")
    
    def benchmark_search(self, ef_search_values=[50, 100, 150, 200, 300], 
                         k=10, num_queries=100):
        """
        基准测试（关键修复：使用独立查询计算真值）
        """
        # 检查索引是否已构建且查询集非空，任一不满足则无法测试，输出错误并返回空结果
        if self.index is None or not self.queries:
            print("Error: Index not built or no queries available")
            return [], []
        
        # 设置随机种子为42，确保每次采样的查询子集可复现，结果一致
        np.random.seed(42)
        # 从独立查询集中随机采样指定数量的查询用于测试
        # min(num_queries, len(self.queries))防止请求数量超过实际可用查询数
        # replace=False确保不放回采样，避免重复选择同一条查询
        sampled_queries = np.random.choice(self.queries, 
                                          min(num_queries, len(self.queries)), 
                                          replace=False)
        
        # 输出开始编码采样查询的提示信息，显示实际采样的查询数量
        print(f"\nEncoding {len(sampled_queries)} independent queries...")
        # 对采样的查询文本进行向量化编码
        query_embeddings = self.model.encode(
            # 将numpy数组转换为列表形式传入编码函数
            list(sampled_queries), 
            # 启用进度条显示编码进度
            show_progress_bar=True, 
            # 指定输出为numpy数组
            convert_to_numpy=True
        ).astype('float32')  # 转换为float32类型以匹配FAISS要求
        # 对查询向量进行L2归一化，确保与文档向量在同一尺度，便于相似度计算
        faiss.normalize_L2(query_embeddings)
        
        # ========== 关键修复：计算真值（Ground Truth）==========
        # 使用暴力搜索（Brute Force）在完整文档库中找到每个查询的真实相关文档作为基准
        print("\nComputing ground truth (Brute Force search in corpus)...")
        print("This finds the truly relevant docs for each user query")
        
        # 创建暴力搜索索引IndexFlatIP（内积），维度与文档向量一致
        # IndexFlatIP通过穷举计算所有内积，保证找到精确最近邻
        brute_index = faiss.IndexFlatIP(self.dim)
        # 将之前保存的语料库向量加载到暴力搜索索引中
        brute_index.add(self.corpus_embeddings)
        
        # 初始化空列表，用于存储每个查询的真实top-k文档ID集合
        ground_truth = []
        # 遍历每个查询向量，使用tqdm显示暴力搜索进度
        for q in tqdm(query_embeddings, desc="Brute Force"):
            # 对单个查询向量进行搜索，reshape(1, -1)将1维向量转为2维(1, dim)以符合FAISS输入要求
            # k参数指定返回最相似的k个文档
            distances, indices = brute_index.search(q.reshape(1, -1), k)
            # 将返回的索引数组转换为Python集合（set），便于后续召回率计算时的交集操作
            # indices[0]取第一行（单个查询的结果），包含k个最相似文档的索引ID
            ground_truth.append(set(indices[0]))
        
        # 初始化空列表，用于存储不同ef_search参数下的测试结果
        results = []
        # 遍历不同的ef_search参数值，测试搜索深度对性能和召回率的影响
        for ef in ef_search_values:
            # 输出当前正在测试的ef_search参数值
            print(f"\nTesting ef_search={ef}...")
            # 动态设置HNSW索引的搜索时探索深度，控制精度和速度的权衡
            self.index.hnsw.efSearch = ef
            
            # 初始化空列表，记录当前参数下每个查询的搜索延迟（毫秒）
            latencies = []
            # 初始化空列表，记录当前参数下每个查询的召回率
            recalls = []
            
            # 同时遍历查询向量和对应的真实文档集合（使用zip配对）
            for q_emb, gt in zip(query_embeddings, ground_truth):
                # 记录搜索开始时间，使用perf_counter获取高精度计时
                start = time.perf_counter()
                # 使用HNSW索引进行近似最近邻搜索，返回k个候选文档
                distances, indices = self.index.search(q_emb.reshape(1, -1), k)
                # 计算搜索耗时，转换为毫秒（乘以1000）
                latency = (time.perf_counter() - start) * 1000
                
                # 将当前查询的延迟添加到列表
                latencies.append(latency)
                
                # 计算召回率：将HNSW检索结果的索引转换为集合
                retrieved = set(indices[0])
                # 计算检索结果与真实值集合的交集大小（命中的相关文档数）
                hits = len(retrieved & gt)
                # 召回率 = 命中数 / k（即检索到的相关文档占应检索总数的比例）
                recalls.append(hits / k)
            
            # 将当前ef_search参数的统计结果封装为字典
            results.append({
                # 当前测试的ef_search参数值
                "ef_search": ef,
                # 平均延迟（毫秒）
                "avg_latency_ms": np.mean(latencies),
                # P95延迟（95百分位数，反映最坏情况表现）
                "p95_latency_ms": np.percentile(latencies, 95),
                # 平均召回率
                "recall_at_k": np.mean(recalls),
                # 最小召回率（最差表现）
                "min_recall": min(recalls),
                # 最大召回率（最好表现）
                "max_recall": max(recalls)
            })
            
            # 输出当前参数的测试结果摘要：平均延迟和召回率范围
            print(f"  Latency: {results[-1]['avg_latency_ms']:.1f}ms | "
                  f"Recall: {results[-1]['recall_at_k']:.1%} "
                  f"(range: {results[-1]['min_recall']:.0%}-{results[-1]['max_recall']:.0%})")
        
        # 返回所有参数组合的测试结果列表，以及实际采样的查询列表（供后续分析使用）
        return results, list(sampled_queries)
    
    def plot_results(self, results):
        """绘制结果（与原版基本相同，增加波动显示）"""
        # 如果结果为空，直接返回，无需绘图
        if not results:
            return
            
        # 从结果字典列表中提取ef_search参数值，用于X轴或标注
        ef_values = [r["ef_search"] for r in results]
        # 提取每个参数对应的平均延迟，用于绘图
        latencies = [r["avg_latency_ms"] for r in results]
        # 提取每个参数对应的平均召回率，用于绘图
        recalls = [r["recall_at_k"] for r in results]
        
        # 创建包含两个子图的绘图区域，横向排列，尺寸为14x5英寸
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 在左侧子图绘制延迟随ef_search变化的曲线图
        ax1.plot(ef_values, latencies, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
        # 添加水平参考线，标记目标延迟阈值50ms，绿色虚线表示
        ax1.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Target <50ms')
        # 设置X轴标签为'ef_search'（搜索深度参数）
        ax1.set_xlabel('ef_search')
        # 设置Y轴标签为'Latency (ms)'（延迟毫秒）
        ax1.set_ylabel('Latency (ms)')
        # 设置图表标题
        ax1.set_title('HNSW: Latency vs Search Depth')
        # 启用网格线，透明度0.3，便于读数但不过于突兀
        ax1.grid(True, alpha=0.3)
        
        # 在右侧子图绘制召回率随延迟变化的曲线图（带误差线显示波动范围）
        ax2.plot(latencies, recalls, 'o-', color='#4ECDC4', linewidth=2, markersize=10)
        # 遍历每个结果点，添加min-max误差线显示召回率的波动范围
        for i, r in enumerate(results):
            # 绘制垂直误差线，从min_recall到max_recall，颜色与主线一致但透明度0.3
            ax2.plot([latencies[i], latencies[i]], 
                    [r['min_recall'], r['max_recall']], 
                    color='#4ECDC4', alpha=0.3, linewidth=3)
        
        # 在每个数据点旁标注对应的ef_search参数值，便于识别
        for i, ef in enumerate(ef_values):
            # 使用annotate添加文本标注，偏移(0,10)像素位于点上方，居中对齐，字号9
            ax2.annotate(f'ef={ef}', (latencies[i], recalls[i]), 
                        xytext=(0,10), textcoords="offset points", 
                        ha='center', fontsize=9)
        
        # 添加水平参考线，标记目标召回率阈值95%，红色虚线表示
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target >95%')
        # 设置X轴标签为'Latency (ms)'（延迟毫秒）
        ax2.set_xlabel('Latency (ms)')
        # 设置Y轴标签为'Recall@10'（前10个结果的召回率）
        ax2.set_ylabel('Recall@10')
        # 设置图表标题，注明包含方差信息
        ax2.set_title('HNSW: Recall vs Latency (with variance)')
        # 添加图例，显示参考线说明
        ax2.legend()
        # 启用网格线
        ax2.grid(True, alpha=0.3)
        
        # 自动调整子图间距，防止标签重叠
        plt.tight_layout()
        # 将图表保存为PNG文件，分辨率150 DPI，文件名包含'_robust'标识修复版本
        plt.savefig('hnsw_benchmark_robust.png', dpi=150)
        # 输出保存成功的提示信息
        print("\n✅ Saved: hnsw_benchmark_robust.png")
        
        # 打印控制台文本报告，使用分隔线增强可读性
        print("\n" + "="*70)
        # 输出报告标题，注明使用真实用户查询对语料库进行测试
        print("HNSW Robust Performance Report (Real User Queries vs Corpus)")
        print("="*70)
        # 遍历所有结果，逐行输出性能指标和状态标记
        for r in results:
            # 判断当前参数是否同时满足延迟<50ms且召回率>95%，满足则标记✅，否则标记⚠️
            status = "✅" if r["avg_latency_ms"] < 50 and r["recall_at_k"] > 0.95 else "⚠️"
            # 格式化输出：状态标记、ef_search值、延迟（保留1位小数）、召回率（百分比，1位小数）
            # 同时显示召回率波动范围（通过平均召回率减最小召回率计算偏差）
            print(f"{status} ef={r['ef_search']:3d} | {r['avg_latency_ms']:5.1f}ms | "
                  f"Recall: {r['recall_at_k']:.1%} (±{r['recall_at_k']-r['min_recall']:.1%})")
        
        # 筛选出同时满足延迟和召回率目标的可行参数配置
        viable = [r for r in results if r["avg_latency_ms"] < 50 and r["recall_at_k"] > 0.95]
        # 如果存在可行配置
        if viable:
            # 从可行配置中选择延迟最小的作为最优推荐
            best = min(viable, key=lambda x: x["avg_latency_ms"])
            # 输出最优参数配置及其性能指标
            print(f"\n🎯 Optimal: ef_search={best['ef_search']} "
                  f"(Recall={best['recall_at_k']:.1%}, Latency={best['avg_latency_ms']:.1f}ms)")
        else:
            # 若无配置同时满足两个目标，给出警告和建议
            print("\n⚠️  未同时满足 <50ms 和 >95%，建议调整M参数或增加ef_search")


# 检查当前脚本是否作为主程序直接运行（而非被导入为模块）
if __name__ == "__main__":
    # 检查必要的输入文件是否存在：语料库文件和查询文件缺一不可
    if not os.path.exists("corpus.jsonl") or not os.path.exists("queries.jsonl"):
        # 若文件缺失，输出错误信息并提示用户先运行准备脚本
        print("Error: corpus.jsonl or queries.jsonl not found!")
        print("Please run prepare_corpus.py first.")
        # 使用exit(1)以非零状态码终止程序，表示执行失败
        exit(1)
    
    # 实例化HNSWBenchmark类，传入语料库和查询文件路径
    # 此时会加载数据并初始化模型
    benchmark = HNSWBenchmark("corpus.jsonl", "queries.jsonl")
    # 调用build_index方法构建HNSW索引，设置M=16和ef_construction=200
    benchmark.build_index(m=16, ef_construction=200)    
    # 执行基准测试，传入ef_search参数列表、top-k值=10、测试查询数=100
    # 返回测试结果列表和实际使用的查询列表（后者暂不需要，用_接收）
    results, _ = benchmark.benchmark_search(
        ef_search_values=[50, 100, 150, 200, 300],
        k=10,
        num_queries=100  # 使用100个真实查询测试
    )
    
    # 调用plot_results方法绘制延迟-召回率曲线并生成性能报告
    benchmark.plot_results(results)
