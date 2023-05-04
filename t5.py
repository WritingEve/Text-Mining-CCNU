from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models import CoherenceModel, LdaModel
from gensim import models
import numpy
#%matplotlib inline


texts = [
            ['苹果','叶子','椭圆形','树上'],
            ['植物','叶子','绿色','落叶乔木'],
            ['水果','苹果','红彤彤','味道'],
            ['苹果','落叶乔木','树上','水果'],
            ['植物','营养','水果','维生素'],
            ['营养','维生素','苹果','成分'],
            ['互联网','电脑','智能手机','高科技'],
            ['苹果','公司','互联网','品质'],
            ['乔布斯','苹果','硅谷'],
            ['电脑','智能手机','苹果','乔布斯'],
            ['苹果','电脑','品质','生意'],
            ['电脑','品质','乔布斯'],
            ['苹果','公司','生意','硅谷']

            ]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


numpy.random.seed(1) # 设置随即种子数，以便相同的设置能跑出相同的结果，可复现
goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary,
     iterations=50, num_topics=2)
badLdaModel = LdaModel(corpus=corpus, id2word=dictionary,
     iterations=50, num_topics=6)


goodcm = CoherenceModel(model=goodLdaModel, corpus=corpus,
       dictionary=dictionary, coherence='u_mass')
badcm = CoherenceModel(model=badLdaModel, corpus=corpus,
       dictionary=dictionary, coherence='u_mass')
print(goodcm.get_coherence())
print(badcm.get_coherence())


goodcm = CoherenceModel(model=goodLdaModel, texts=texts,
    dictionary=dictionary,  coherence='c_v')
badcm = CoherenceModel(model=badLdaModel, texts=texts,
    dictionary=dictionary,  coherence='c_v')
print(goodcm.get_coherence())
print(badcm.get_coherence())