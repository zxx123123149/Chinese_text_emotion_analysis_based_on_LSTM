#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
df=pd.read_csv('train1.csv')


# In[ ]:


df.sample(10)


# In[3]:



print("在 Text 列中总共有 %d 个空值." % df['Text'].isnull().sum())
print("在 lable列中总共有 %d 个空值." % df['label'].isnull().sum())
df[df.isnull().values==True]
df = df[pd.notnull(df['label'])]


# In[4]:


d = {'label':df['label'].value_counts().index, 'count': df['label'].value_counts()}
df_label = pd.DataFrame(data=d).reset_index(drop=True)
df_label


# In[5]:


import os
import matplotlib.pyplot as plt
df_label.plot(x='label', y='count', kind='bar', legend=False,  figsize=(8, 5))
plt.ylabel('count', fontsize=20)
plt.xlabel('label', fontsize=20)


# In[6]:


df['label_id'] = df['label'].factorize()[0]
label_id_df = df[['label', 'label_id']].drop_duplicates().sort_values('label_id').reset_index(drop=True)
label_to_id = dict(label_id_df.values)
id_to_label = dict(label_id_df[['label_id', 'label']].values)
df.sample(10)


# In[7]:


label_id_df


# In[8]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.preprocessing import text
# 设置最频繁使用的50000个词
MAX_NB_WORDS = 50000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 30
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100
 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))


# In[9]:


X = tokenizer.texts_to_sequences(df['Text'].values)
#填充X,让X的各个列的长度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
 
#多类标签的onehot展开
Y = pd.get_dummies(df['label_id']).values
 
print(X.shape)
print(Y.shape)


# In[10]:


#拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[11]:


#定义模型
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[12]:


epochs = 3
batch_size = 64
 
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[13]:


from  sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix 
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis = 1)
Y_test = Y_test.argmax(axis = 1)
print('accuracy %s' % accuracy_score(y_pred, Y_test))
print(classification_report(Y_test, y_pred,target_names=label_id_df['label'].values))


# In[35]:


import jieba as jb
import re
#定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

#加载停用词
stopwords = stopwordslist("chineseStopWords.txt.txt")
def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    label_id= pred.argmax(axis=1)[0]
    return label_id_df[label_id_df.label_id==label_id]['label'].values[0]


# In[36]:


predict('我真的很开心呢！')


# In[ ]:




