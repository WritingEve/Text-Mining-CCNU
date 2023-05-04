import logging
import tempfile
import os.path
from gensim import corpora, models, similarities
from collections import defaultdict
import jieba

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEMP_FOLDER = tempfile.gettempdir()
print('文件夹"{}" 将被用来存储语料和临时性的字典'.format(TEMP_FOLDER))

if os.path.isfile(os.path.join(TEMP_FOLDER, 'deerwester.dict')):
    dictionary = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'deerwester.dict'))
    corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'deerwester.mm'))
    print("使用前面教程中产生的语料文件。")
else:
    print("请运行前面的教程，以生成语料文件。")

jieba.add_word('知识图谱')
raw_corpus = ['商业新知:知识图谱为内核,构建商业创新服务完整生态。',
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

raw_corpus = [' '.join(jieba.lcut(i)) for i in raw_corpus]

stoplist = [i.strip() for i in open('datasets/stopwords_zh.txt', encoding='utf-8').readlines()]

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(processed_corpus)
dictionary.token2id

#查看保存字典（Dictionary）中的前3个词汇：
print(dictionary[0])
print(dictionary[1])
print(dictionary[2])

tfidf = models.TfidfModel(corpus)

doc_bow = [(0, 3), (1, 5)]
print(tfidf[doc_bow])

#或者，直接对整个（训练）语料库进行特征转换：
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3) # 初始化 LSI 转换
corpus_lsi = lsi[corpus_tfidf] #在原始语料词袋表示的基础上创建一个双包装器（Double Wrapper）：bow-> tfidf-> lsi

lsi.show_topics()

for doc in corpus_lsi: # bow->tfidf转换 和 tfidf->lsi转换实际上是在这里即时完成的
    print(doc)

lsi.save(os.path.join(TEMP_FOLDER, 'model.lsi'))
lsi = models.LsiModel.load(os.path.join(TEMP_FOLDER, 'model.lsi'))