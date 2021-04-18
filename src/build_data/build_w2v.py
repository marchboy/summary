#-*- coding:utf-8 -*-

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


import sys
sys.path.append("D:/Projects/TextSummary/week2_项目导论中与中文词向量实践/summary/")

from src.utils import config
from src.utils.multi_proc_utils import cores


def build_w2v():
    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(sentences=LineSentence(config.merger_seg_path),
                        # vector_size=config.embedding_dim,
                        size=config.embedding_dim,
                        sg=1,
                        workers=cores,
                        # epochs=config.wv_train_epochs,
                        iter=config.wv_train_epochs,
                        window=5,
                        min_count=5)
    return wv_model
