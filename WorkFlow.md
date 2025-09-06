# 基于局部敏感哈希的新闻洗稿检测系统的设计与实现
***Design and Implementation of a Journalistic Content Reuse Detection System Using LSH***

## 创建环境
1. 建立文档格式
GRADUATION/  # 项目根目录
> page/                     # 主要Python代码包
> > __init__.py           # 让Python将app视为一个包（package）
> > main.py               # FastAPI应用的入口文件
> > models.py             # 定义数据模型（Pydantic和数据库模型）
> > database.py           # 数据库连接和操作
> > sentence_encoder.py   # SBERT编码器（核心AI模块）
> > lsh_index.py          # LSH索引和搜索逻辑（核心算法模块）
> > file_processor.py     # 处理PDF、DOCX等文件上传和文本提取
> > text_aligner.py       # 负责句对对齐和高亮逻辑
> > routers/              # 存放所有API路由的包
> > > __init__.py         # 让Python将routers视为一个包
> > > articles.py         # 与文章相关的路由（上传、入库、管理）
> > > detection.py        # 与查重相关的路由（单篇、双篇）
> > > reports.py          # 与报告相关的路由（查看、删除、导出）
> requirements.txt        # 项目依赖的Python库清单
> README.md              # 项目说明文档

2. 创建并填写 requirements.txt 文件
在项目根目录下创建 requirements.txt 文件。这个文件至关重要，它告诉任何想运行你项目的人需要安装哪些库。

3. 创建虚拟环境并安装依赖
在终端中，逐行输入并执行以下命令：
# 1. 创建虚拟环境（会在项目根目录生成一个 `venv` 文件夹）
python -m venv venv

# 2. 激活虚拟环境
#    对于 Windows:
.\venv\Scripts\activate
#    对于 macOS/Linux:
source venv/bin/activate

# 激活成功后，终端命令行的前面应该会出现 `(venv)` 字样。
# 像这样: (venv) PS C:\...\news_plagiarism_detection>

# 3. 使用pip安装requirements.txt里列出的所有依赖
#    (确保终端中的Python是虚拟环境里的，而不是系统的)
pip install -r requirements.txt
pip list

