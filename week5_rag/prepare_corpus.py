from datasets import load_dataset
import json


def prepare_corpus(target_docs=50000, target_queries=1000):
    """
    准备语料库和独立查询集
    
    Returns:
        tuple: (文档列表, 查询列表)
    """
    # ========== 1. 准备文档库（Wikipedia）==========
    # 在控制台输出提示信息，告知用户开始加载Wikipedia语料
    print("Loading Wikipedia snippets for corpus...")
    
    # 使用Hugging Face datasets库加载流式数据集
    # "wiki_snippets"是数据集名称，"wiki40b_en_100_0"是配置名称
    # streaming=True启用流式加载，避免一次性加载全部数据到内存
    # split="train"指定使用训练集部分
    ds = load_dataset("wiki_snippets", "wiki40b_en_100_0", streaming=True, split="train")
    
    # 初始化空列表，用于存储从数据集中提取的文档内容
    corpus = []
    
    # 使用enumerate遍历数据集流，i为索引，item为数据条目
    for i, item in enumerate(ds):
        # 检查是否已达到目标文档数量，达到则停止采集
        if i >= target_docs:
            break
        
        # 构建文档文本：将文章标题和正文段落组合
        # 格式为"Title: {标题}\n{正文}"，便于后续检索时关联标题与内容
        text = f"Title: {item['article_title']}\n{item['passage_text']}"
        
        # 将构建好的文档文本添加到语料库列表中
        corpus.append(text)
        
        # 每加载10000条文档输出一次进度信息，便于监控长时间运行的任务
        if i % 10000 == 0 and i > 0:
            print(f"Loaded {i} docs")
    
    # 以写入模式打开corpus.jsonl文件，使用UTF-8编码支持多语言字符
    with open("corpus.jsonl", "w", encoding="utf-8") as f:
        # 遍历语料库中的每一条文档
        for doc in corpus:
            # 将文档封装为字典结构，键为"text"，值为文档内容
            # 使用json.dumps序列化为JSON字符串，ensure_ascii=False保留非ASCII字符原样
            # 每条JSON记录后添加换行符，形成JSON Lines格式（每行一条独立JSON）
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    
    # 计算语料库总字符数，转换为MB（除以1024*1024）
    # sum(len(d) for d in corpus)计算所有文档的字符总数
    total_size = sum(len(d) for d in corpus) / 1024 / 1024
    
    # 输出语料库统计信息：文档数量和总大小（保留1位小数）
    print(f"✅ Corpus: {len(corpus)} docs, {total_size:.1f}MB")
    
    # ========== 2. 准备独立查询集（MS MARCO真实用户查询）==========
    # 输出提示信息，告知用户开始加载MS MARCO查询数据集
    print("\nLoading MS MARCO real user queries...")
    
    # 加载MS MARCO数据集的验证集，这是真实的用户搜索查询数据
    # 使用流式加载避免内存溢出，split="validation"指定验证集（通常包含真实用户查询）
    queries_ds = load_dataset("ms_marco", "v1.1", split="validation", streaming=True)
    
    # 初始化空列表，用于存储查询文本
    queries = []
    
    # 遍历MS MARCO查询数据集，i为索引，item为查询条目
    for i, item in enumerate(queries_ds):
        # 检查是否已达到目标查询数量
        if i >= target_queries:
            break
        
        # 从数据条目中提取查询文本字段
        # "query"字段包含用户的真实搜索问题
        query_text = item["query"]
        
        # 将查询文本添加到查询列表
        queries.append(query_text)
        
        # 每加载500条查询输出一次进度信息
        if i % 500 == 0 and i > 0:
            print(f"Loaded {i} queries")
    
    # 以写入模式打开queries.jsonl文件，同样使用UTF-8编码
    with open("queries.jsonl", "w", encoding="utf-8") as f:
        # 遍历所有查询文本
        for q in queries:
            # 将每条查询封装为字典并序列化为JSON字符串
            # 同样使用ensure_ascii=False保持字符原样
            # 每条记录后添加换行符，符合JSON Lines格式规范
            f.write(json.dumps({"text": q}, ensure_ascii=False) + "\n")
    
    # 输出查询集统计信息，确认成功保存的查询数量
    print(f"✅ Queries: {len(queries)} independent user queries")
    
    # 返回包含语料库列表和查询列表的元组，供调用方后续使用
    return corpus, queries


# 判断当前脚本是否作为主程序直接运行（而非被导入为模块）
if __name__ == "__main__":
    # 调用prepare_corpus函数，传入默认参数：50000篇文档和1000条查询
    # 开始执行语料库和查询集的构建与保存流程
    prepare_corpus(50000, 1000)
