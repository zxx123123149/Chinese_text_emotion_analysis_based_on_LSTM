# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import linecache
import copy
import json
import jieba

class ReadEmbeddings():

    def __init__(self, emb_file='D:\作业资料存放区\我的毕设项目__3代\第3代_部分保存\SRNN部分__部分内容\embeddings__部分\Tencent_AILab_ChineseEmbedding.txt',
                 map_file='D:\作业资料存放区\我的毕设项目__3代\第3代_部分保存\SRNN部分__部分内容\embeddings__部分\embeddings_map_index.txt', \
                 encoding='utf8'):
        self.map_file = map_file
        self.emb_file = emb_file
        self.encoding = encoding
        self.counter = 0
        self.max_count = 8824330

    def load_map_in_memery(self):
        '''
        因为词嵌入文件过大，所以我们只需要加载它的映射就可以了

        返回：
            map_list -- 映射列表
        '''
        # 将映射表加载进内存
        tmp_map_list = linecache.getlines(self.map_file)
        map_list = []
        for m in tmp_map_list:
            curr_m = m.split()
            curr_list_word = [curr_m[0]]
            curr_list_value = list(map(int, curr_m[1:]))
            map_list.append(curr_list_word + curr_list_value)
        return map_list

    def clear_cache(self):
        linecache.clearcache()

    def find_by_list(self, map_list, query_list, f_log_obj=None):
        '''
        批量查询它的词嵌入权值
        参数：
            map_list -- 映射列表，可通过load_map_in_memery()函数得到。
            query_list -- 要查询的词汇的列表
            defalut -- 默认值【"0" | "random"】
        返回：
            return_dict -- 字典类型，键为词汇，值为对应的词嵌入权值。
                example：{'我们':[0.238955, -0.192848, ... , 0.137744],'...':[...],...}
        '''
        is_log = False
        if f_log_obj:
            is_log = True

        query_list2 = copy.deepcopy(query_list)
        if len(query_list2) == 0:
            if is_log:
                is_log = False
                f_log_obj.write("query_list is empty!\n")
            return -1

        return_dict = {}
        with open(self.emb_file, 'rb') as emb:
            for m in map_list:
                for q in query_list2:
                    if q in m:
                        emb.seek(m[2])
                        value = list(map(float, emb.read(m[3]).split()[1:]))
                        return_dict[str(q)] = value
                        query_list2.remove(q)
        if len(query_list2) >= 1:
            for q in query_list2:
                # print("Waring: " + q + " not in the embeddings.")
                if is_log:
                    f_log_obj.write("未找到：{word} \n".format(word=str(q)))
                return_dict[str(q)] = [0.0] * 200
                query_list2.remove(q)

        return return_dict

    def find_by_word(self, map_list, word, default_value):
        with open(self.emb_file, 'rb') as emb:
            for m in map_list:
                if word in m:
                    emb.seek(m[2])
                    value = list(map(float, emb.read(m[3]).split()[1:]))
#                    for i in range(len(value)):
#                        if value[i] < 0:
#                            value[i] = -value[i]
                    #if is_del_element:
                    #    map_list.remove(m)
                        # f_log_obj.write("删除：{m} \n".format(m=str(m)))
                    return value
            value = default_value
            return value


from keras.preprocessing import text
df=pd.read_csv('processed_set.csv',encoding='utf-8')
print(df.sample(10))

print("在 Text 列中总共有 %d 个空值." % df['Text'].isnull().sum())
print("在 lable列中总共有 %d 个空值." % df['label'].isnull().sum())
df[df.isnull().values == True]
df = df[pd.notnull(df['label'])]

d = {'label': df['label'].value_counts().index, 'count': df['label'].value_counts()}
df_label = pd.DataFrame(data=d).reset_index(drop=True)
df_label



df['label_id'] = df['label'].factorize()[0]
label_id_df = df[['label', 'label_id']].drop_duplicates().sort_values('label_id').reset_index(drop=True)
label_to_id = dict(label_id_df.values)
id_to_label = dict(label_id_df[['label_id', 'label']].values)

print(df.sample(10))

print(label_id_df)


import tensorflow
from keras.preprocessing import text
# 设置最频繁使用的50000个词
MAX_NB_WORDS = 37864

# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 80
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 200

tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))
# ****
def save_dict(filename, dic):
    '''save dict into json file'''

    with open(filename, 'a+', encoding="utf8") as json_file:
        json.dump(dic, json_file, ensure_ascii=False)


save_dict("word_index_1.json", word_index)
# ****


from keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(df['Text'].values)
# 经过上一步操作后，X为整数构成的两层嵌套list
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# 经过上步操作后，此时X变成了numpy.ndarray
# 多类标签的onehot展开
Y = pd.get_dummies(df['label_id']).values

print(X.shape)
print(Y.shape)


from sklearn.model_selection import train_test_split
#拆分训练集和测试集，X为被划分样本的特征集，Y为被划分样本的标签
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
model = Sequential()
#*********************************
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

EMBEDDING_DIM = 200
emb = ReadEmbeddings()
emb_map_list = emb.load_map_in_memery()
embedding_matrix = np.random.random((MAX_NB_WORDS+1, EMBEDDING_DIM))
for word, i in word_index.items():
    query_word = word.replace("'", "").strip()
    embedding_vector = emb.find_by_word(emb_map_list, query_word, default_value=[0.0]*200)
    if embedding_vector != [0.0]*200:
        embedding_matrix[i] = embedding_vector
#embedding_layer = tf.keras.layers.Embedding(MAX_NB_WORDS,
#                                            EMBEDDING_DIM,
#                                            weights=[embedding_matrix],
#                                            input_length=MAX_SEQUENCE_LENGTH,
#                                            trainable=True)
#*********************************

#model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(tensorflow.keras.layers.Embedding(MAX_NB_WORDS+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH)) #,weights=[embedding_matrix]
model.add(SpatialDropout1D(0.2))#dropout会随机独立地将部分元素置零，而SpatialDropout1D会随机地对某个特定的纬度全部置零
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))#输出层包含10个分类的全连接层，激活函数设置为softmax
opt = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #使用adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

# 保存在验证集中效果最好的模型
savebestmodel = 'LSTM.h5'
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(savebestmodel, monitor='accuracy', verbose=1, save_best_only=True,
                                                mode='max')
#callbacks = [checkpoint]

from keras.callbacks import EarlyStopping
epochs = 25
batch_size = 64 #指定梯度下降时每个batch包含的样本数
#callbacks（list），其中元素是keras.callbacks.Callback的对象。这个list的回调函数将在训练过程中的适当时机被调用
#validation_split指定训练集中百分之十的数据作为验证集
history = model.fit(X_train, Y_train, epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[checkpoint],verbose = 1)
#                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])



accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

f = open("out.txt", "w")
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]), file = f)




