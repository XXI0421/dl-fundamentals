import matplotlib.pyplot as plt
import numpy as np

# 仓库信息
repo_info = [
    {'name': 'public-apis/public-apis', 'stars': 421780, 'forks': 45937, 'updated_at': '2026-04-14'},
    {'name': 'EbookFoundation/free-programming-books', 'stars': 385531, 'forks': 66106, 'updated_at': '2026-04-14'},
    {'name': 'donnemartin/system-design-primer', 'stars': 342771, 'forks': 55383, 'updated_at': '2026-04-14'},
    {'name': 'vinta/awesome-python', 'stars': 292298, 'forks': 27671, 'updated_at': '2026-04-14'},
    {'name': 'TheAlgorithms/Python', 'stars': 219601, 'forks': 50328, 'updated_at': '2026-04-14'},
    {'name': 'Significant-Gravitas/AutoGPT', 'stars': 183425, 'forks': 46219, 'updated_at': '2026-04-14'},
    {'name': 'AUTOMATIC1111/stable-diffusion-webui', 'stars': 162366, 'forks': 30273, 'updated_at': '2026-04-14'},
    {'name': 'huggingface/transformers', 'stars': 159373, 'forks': 32869, 'updated_at': '2026-04-14'},
    {'name': 'yt-dlp/yt-dlp', 'stars': 156849, 'forks': 12916, 'updated_at': '2026-04-14'},
    {'name': '521xueweihan/HelloGitHub', 'stars': 151107, 'forks': 11506, 'updated_at': '2026-04-14'}
]

# 提取项目名称和star数量
names = [repo['name'] for repo in repo_info]
stars = [repo['stars'] for repo in repo_info]

# 创建图表
fig, ax = plt.subplots()
ax.barh(names, stars)
ax.set_xlabel('Stars')
ax.set_title('Top 10 Python Repositories on GitHub')
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.xticks(rotation=90)
plt.tight_layout()

# 保存图表为图片
plt.savefig('python_repos_trend.png')

# 保存图表为文本
plt.savefig('python_repos_trend.txt')