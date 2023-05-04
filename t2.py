import logging
import os
import tempfile
from collections import defaultdict
from gensim import corpora
import jieba
from gensim.corpora import dictionary
from smart_open import smart_open
from pprint import pprint
from six import iteritems
from smart_open import smart_open
import gensim
import numpy as np
import scipy.sparse


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEMP_FOLDER = tempfile.gettempdir()
print('文件夹"{}" 将被用来存储语料和临时性的字典'.format(TEMP_FOLDER))

jieba.add_word('知识图谱') #防止“知识图谱”被切错词

docs = ['商业新知:知识图谱为内核,构建商业创新服务完整生态。',
'如何更好利用知识图谱技术做反欺诈? 360金融首席数据科学家沈赟开讲。',
'知识管理 | 基于知识图谱的国际知识管理领域可视化分析。',
'一文详解达观数据知识图谱技术与应用。',
'知识图谱技术落地金融行业的关键四步。',
'一文读懂知识图谱的商业应用进程及技术背景。',
'海云数据CPO王斌:打造大数据可视分析与AI应用的高科技企业。',
'智能产业|《人工智能标准化白皮书2018》带来创新创业新技术标准。',
'国家语委重大科研项目“中华经典诗词知识图谱构建技术研究”开题。',
'最全知识图谱介绍:关键技术、开放数据集、应用案例汇总。',
'中译语通Jove Mind知识图谱平台 引领企业智能化发展。',
'知识图谱:知识图谱赋能企业数字化转型，为企业升级转型注入新能量。']

documents = [' '.join(jieba.lcut(i)) for i in docs]
print(documents)

# 移除常用词以及分词
stoplist = [i.strip() for i in open('datasets/stopwords_zh.txt', encoding='utf-8').readlines()]
texts = [[word for word in document.lower().split() if word not in stoplist]
for document in documents]


# 移除仅出现一次的词汇
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1


texts = [[token for token in text if frequency[token] > 1] for text in texts]


pprint(texts) #使打印的格式更齐整


dictionary = corpora.Dictionary(texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # 保存字典，以备后续查找之用
print(dictionary)
print(dictionary.token2id)

new_doc = "知识图谱 为 企业 转型 助力"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  #保存到本地，以作后用
for c in corpus:
    print(c)

class MyCorpus(object):
    def __iter__(self):
        for line in smart_open('datasets/mycorpus.txt', 'r',encoding='utf-8'):
        # 假设每一行一个文档，用jieba进行分词
            yield dictionary.doc2bow(' '.join(jieba.lcut(line)).lower().split())

corpus_memory_friendly = MyCorpus()  #不需要将语料载入到内存中!
print(corpus_memory_friendly)

for vector in corpus_memory_friendly:  #每次载入一个文档向量
    print(vector)

#收集所有词汇的统计信息
dictionary = corpora.Dictionary(' '.join(jieba.lcut(line)).lower().split() for line in smart_open('datasets/mycorpus.txt', 'r',encoding='utf-8'))

#停用词和低频词（这里指仅出现1次的词汇）的ID集合
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

#真正实施去停用词和低频次的操作
dictionary.filter_tokens(stop_ids + once_ids)
print(dictionary)

#创建一个包含2个文档的微小语料，以一个python列表呈现

corpus = [[(1, 0.5)], []]  # 其中一个文档故意搞成空的

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.mm'), corpus)

corpora.SvmLightCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.svmlight'), corpus)
corpora.BleiCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.lda-c'), corpus)
corpora.LowCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.low'), corpus)

corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'corpus.mm'))

# 一种打印语料库的方式是 --- 将其整个载入内存中
print(list(corpus))  #  调用 list() 能将任何序列转化为普通的Python list


# 另一种方法：一次打印一个文档
for doc in corpus:
    print(doc)

corpora.BleiCorpus.serialize(os.path.join(TEMP_FOLDER, 'corpus.lda-c'), corpus)
numpy_matrix = np.random.randint(10, size=[5,2])
print(numpy_matrix)

corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
numpy_matrix_dense = gensim.matutils.corpus2dense(corpus, num_terms=10)
print(numpy_matrix_dense)

scipy_sparse_matrix = scipy.sparse.random(5,2)
print(scipy_sparse_matrix)

corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
print(corpus)

scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
print(scipy_csc_matrix)