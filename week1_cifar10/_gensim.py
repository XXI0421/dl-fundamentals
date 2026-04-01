import gensim.downloader

# 加载模型
model = gensim.downloader.load('glove-wiki-gigaword-50')

# 1. 查找与 "tower" 最相似的10个词
print(model.most_similar('tower', topn=10))

# 2. 经典的词类比：king - man + woman = queen
result1 = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print("king - man + woman ≈", result1)

result2 = model.most_similar(positive=['uncle', 'woman'], negative=['man'], topn=3)
print("uncle - man + woman ≈", result2)