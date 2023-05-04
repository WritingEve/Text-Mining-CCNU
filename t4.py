import os
import tempfile
from pprint import pprint
import logging
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint  #使打印的格式更齐整
import jieba


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TEMP_FOLDER = tempfile.gettempdir()
print('文件夹"{}" 将被用来存储语料和临时性的字典'.format(TEMP_FOLDER))

#对特定长词进行控制，防止被分错词，影响后续的分析效果
jieba.add_word('微信')
jieba.add_word('文本挖掘')
jieba.add_word('增长黑客')
jieba.add_word('小白')
jieba.add_word('大数据')


docs = [
'数据挖掘实操｜用文本挖掘剖析近5万首《全唐诗》',
'以虎嗅网4W+文章的文本挖掘为例，展现数据分析的一整套流程',
'干货｜作为一个合格的“增长黑客”，你还得重视外部数据的分析！',
'文本挖掘从小白到精通（二）---语料库和词向量空间',
'文本挖掘从小白到精通（三）---主题模型和文本数据转换',
'文本挖掘从小白到精通（一）---语料、向量空间和模型的概念',
'以《大秦帝国之崛起》为例，来谈大数据舆情分析和文本挖掘',
'文本分类算法集锦，从小白到大牛，附代码注释和训练语料',
'Social Listening和传统市场调研的关系是怎样的？',
'【新媒体运营实操】如何做出一个精美的个性化词云'
'以哈尔滨冰雪大世界旅游的传播效应为例，谈数据新闻可视化的“魅惑”',
'万字干货｜10款数据分析“工具”，助你成为新媒体运营领域的“增长黑客”',
'如何用数据分析，搞定新媒体运营的定位和内容初始化？',
'当数据分析遭遇心理动力学：用户深层次的情感需求浮出水面',
'揭开微博转发传播的规律：以“人民日报”发布的G20文艺晚会微博为例',
'数据运营实操 | 如何运用数据分析对某个试运营项目进行“无死角”的复盘？',
'如何利用微信后台数据优化微信运营',
'如何利用Social Listening从社会化媒体中“提炼”有价值的信息？',
'用大数据文本挖掘，来洞察“共享单车”的行业现状及走势',
'从社交媒体传播和文本挖掘角度解读《欢乐颂2》',
'不懂数理和编程，如何运用免费的大数据工具获得行业洞察？',
'写给迷茫的你：如何运用运营思维规划自己的职业发展路径？',
'如何用聚类分析进行企业公众号的内容优化',
'傅园慧和她的“洪荒之力”的大数据舆情分析',
'数据运营|数据分析中，文本分析远比数值型分析重要！（上）'
        ]
#再对文本进行分词，用空格隔开组成字符串，方便进行下一步的处理
documents = [' '.join(jieba.lcut(i)) for i in docs]
documents

# 去停用词
stoplist = [i.strip() for i in open('datasets/stopwords_zh.txt',encoding='utf-8').readlines()]
texts = [[word for word in document.lower().split() if word not in stoplist]
for document in documents]

pprint(texts)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 查看词袋表示中的部分数据
list(corpus)[:3]


lsi = models.LsiModel(
         corpus,
         id2word=dictionary,
         power_iters=100,
         num_topics=10
         )

#查询语句为“文本挖掘在舆情口碑挖掘中的作用很大”
doc = "文本挖掘 在 舆情 口碑 挖掘 中 的 作用 很大 "
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]      #将查询语句转换到LSI向量空间


result = [(docs[i[0]],i[1]) for i in vec_lsi]
pprint(sorted(result,key=lambda x: x[1],reverse=True))

#index = similarities.MatrixSimilarity(lsi[corpus])  #将查询语料库转换到LSI向量空间并对其中的每个文档/语句建立索引
#内存友好型接口
index = similarities.Similarity(output_prefix='Similarity',corpus=lsi[corpus],num_features=500)  #将查询语料库转换到LSI向量空间并对其中的每个文档/语句建立索引

index.save(os.path.join(TEMP_FOLDER, '查询.index'))
index = similarities.MatrixSimilarity.load(os.path.join(TEMP_FOLDER, '查询.index'))

sims = index[vec_lsi]
result = [(docs[i[0]],i[1]) for i in enumerate(sims)]            # 对检索语料库进行相似度查询
pprint(sorted(result ,key=lambda x: x[1],reverse=True))          # 每个查询结果的格式是 (语句， 与查询语句的相似度) ，是一个2元元组