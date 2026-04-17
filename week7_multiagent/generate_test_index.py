"""
测试索引生成脚本 - 为Week 7多Agent系统生成测试数据
"""
import shutil
from pathlib import Path
from memory.long_term import get_long_term_memory, LongTermMemory, _sessions


def reset_storage():
    """清理存储，生成新索引"""
    storage = Path("./long_term_memory")
    if storage.exists():
        shutil.rmtree(storage)
    
    # 重置全局会话缓存
    _sessions.clear()


def generate_test_facts():
    """生成测试事实数据"""
    ltm = get_long_term_memory()
    
    # 用户画像数据
    user_facts = [
        {"text": "用户偏好Python语言而非JavaScript", "category": "user_profile", "importance": 0.9},
        {"text": "用户喜欢使用PyGame进行游戏开发", "category": "user_profile", "importance": 0.8},
        {"text": "用户希望开发一个2D俯视角游戏", "category": "user_profile", "importance": 0.85},
        {"text": "用户对UI美观度有较高要求", "category": "user_profile", "importance": 0.7},
    ]
    
    # SOP进度数据
    sop_facts = [
        {"text": "ProductManager完成了game_design.md，产出摘要：设计了一个2D俯视角RPG游戏", "category": "sop_progress", "importance": 0.9},
        {"text": "Architect完成了tech_architecture.md，产出摘要：采用MVC架构模式", "category": "sop_progress", "importance": 0.85},
        {"text": "SoftwareEngineer完成了prototype_v1.py，产出摘要：实现了基础游戏循环", "category": "sop_progress", "importance": 0.8},
    ]
    
    # 技术决策数据
    tech_facts = [
        {"text": "技术栈选择：Pygame + Python", "category": "tech_decision", "importance": 0.9},
        {"text": "架构：MVC模式", "category": "tech_decision", "importance": 0.85},
        {"text": "数据库：SQLite（轻量级）", "category": "tech_decision", "importance": 0.7},
        {"text": "Pygame仅支持2D，不支持3D功能", "category": "tech_decision", "importance": 0.95},
    ]
    
    # 项目信息
    project_facts = [
        {"text": "项目名称：Fantasy RPG", "category": "general", "importance": 0.8},
        {"text": "目标平台：PC（Windows/Linux）", "category": "general", "importance": 0.7},
        {"text": "预计开发周期：3个月", "category": "general", "importance": 0.6},
    ]
    
    # 添加所有事实
    all_facts = user_facts + sop_facts + tech_facts + project_facts
    
    for fact in all_facts:
        ltm.add_fact(
            text=fact["text"],
            category=fact["category"],
            importance=fact["importance"]
        )
    
    print(f"\n✅ 测试索引生成完成！")
    print(f"   总事实数: {ltm.get_fact_count()}")
    print(f"   存储位置: ./long_term_memory")
    print(f"   会话ID: {ltm.session_id}")
    
    return ltm


def test_retrieval(ltm):
    """测试检索功能"""
    print("\n" + "="*50)
    print("测试检索功能")
    print("="*50)
    
    queries = [
        "游戏设计文档",
        "技术架构",
        "用户偏好",
        "3D引擎",
        "Python",
    ]
    
    for query in queries:
        results = ltm.retrieve(query, top_k=3)
        print(f"\n📖 查询: '{query}'")
        if results:
            for i, res in enumerate(results):
                print(f"   {i+1}. [{res['category']}] {res['text'][:50]}... (相似度: {res['score']:.3f})")
        else:
            print(f"   无匹配结果")


if __name__ == "__main__":
    print("🧠 生成Week 7测试索引")
    print("="*50)
    
    # 重置存储并生成测试数据
    reset_storage()
    ltm = generate_test_facts()
    
    # 测试检索功能
    test_retrieval(ltm)
    
    print("\n🎉 测试索引生成完毕！")
