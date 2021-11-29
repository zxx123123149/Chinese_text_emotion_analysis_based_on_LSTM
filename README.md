# Chinese_text_emotion_analysis_based_on_LSTM
# 基于LSTM网络的中文文本情感分析任务
# １　摘要
本项目基于LSTM（长短期记忆网络）来实现对中文文本情感的分类及预测。本实验采用高质量的六情感微博数据，利用Jieba进行分词，将中文文本序列化，同时进行词嵌入，再使用LSTM算法进行核心的分类器训练，调整模型实现最优的准确率。最后自定义一个预测函数，通过输入一段中文文本来预测文本的情感。

This project is based on LSTM (Short- and Short-Term Memory Network) to classify and predict Chinese text emotions. This experiment uses high-quality six-emotion micro-blog data, uses Jieba to serialize Chinese text, embeds words, and then uses LSTM algorithm to train the core classifier, adjust the model to achieve optimal accuracy. Finally, customize a prediction function to predict the emotion of the text by entering a piece of Chinese text.

# ２数据介绍及预处理工作
数据包含两个维度，一是中文的文本内容，主要来自微博上的对话内容，第二个维度是根据对话内容所打的情感标签，主要包含６大类，分别是Ｈａｐｐｉｎｅｓｓ，Ｌｏｖｅ，Ｓｏｒｒｏｗ
