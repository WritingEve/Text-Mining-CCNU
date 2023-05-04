# 在命令行中（我是在anaconda prompt）中输入命令 pip install jieba以及 pip install gensim
import jieba
from gensim import corpora
from gensim import models
from collections import defaultdict


#jieba.add_word()表示向分词词典中增加新词，此处是防止知识图谱被切错词
jieba.add_word('知识图谱')


# 将12个文档中的语句放入列表中
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

#jieba.lcut()返回精确模式，输出的分词能够完整且不多余地组成原始文本
#' '.join(jieba.lcut(i))是以空格作为分隔符，将jieba.lcut(i)所有的元素合并成一个新的字符串
raw_corpus = [' '.join(jieba.lcut(i)) for i in raw_corpus]
#print(raw_corpus)

# 移除常用词以及分词
stoplist = [i.strip() for i in open('datasets/stopwords_zh.txt',encoding='utf-8').readlines()]


#将文档中可能存在的西文字符小写化，按空格进行拆分，且去停用词
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]
#相当于代码
#texts=[]
#for document in raw_corpus:
#    words=[]
#    for word in document.lower().split():
#        if word not in stoplist():
#            words.append(word)
#texts.append(words)

#计算词频
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1


# 仅保留词频数高于1的词汇
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
#processed_corpus=[]
#for text in texts:
#	 tokens=[]
#	 for token in text:
#		 if frequency[token]>1:
#			 tokens.append(token)
#processed_corpus(tokens)
#print(processed_corpus)

#使用gensim.corpora.Dictionary完成，将语料库中的每个词汇与唯一的整数ID相关联
dictionary = corpora.Dictionary(processed_corpus)
#print(dictionary)


#dictionary.token2id是一个字典，它将词语映射到一个唯一的整数ID。这些数字ID可以用于训练模型、计算文本相似度等任务
dictionary.token2id
#print(dictionary.token2id)

new_doc = "知识图谱 为 企业 转型 助力"  #已分词，便于后续处理
new_vec = dictionary.doc2bow(new_doc.lower().split())
#print(new_vec)

#将整个原始语料库转换为向量列表:
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus] #processed_corpus 是处理后词频数高于1的词汇
#print(bow_corpus)

# 训练模型
tfidf = models.TfidfModel(bow_corpus) #bow_corpus是原始语料库转化成的向量列表
# 对"知识图谱这种技术是企业转型的利器"进行转换
print(tfidf[dictionary.doc2bow("知识图谱 这种 技术 是 企业 转型 的 利器".split())]) #dictionary的`doc2bow`方法为该语句创建词袋表示，该方法返回词汇计数的稀疏表示
#print(tfidf)