# Chinese_text_emotion_analysis_based_on_LSTM
# 基于LSTM网络的中文文本情感分析任务
# １　摘要
本项目基于LSTM（长短期记忆网络）来实现对中文文本情感的分类及预测。本实验采用高质量的六情感微博数据，利用Jieba进行分词，将中文文本序列化，同时进行词嵌入，再使用LSTM算法进行核心的分类器训练，调整模型实现最优的准确率。最后自定义一个预测函数，通过输入一段中文文本来预测文本的情感。

This project is based on LSTM (Short- and Short-Term Memory Network) to classify and predict Chinese text emotions. This experiment uses high-quality six-emotion micro-blog data, uses Jieba to serialize Chinese text, embeds words, and then uses LSTM algorithm to train the core classifier, adjust the model to achieve optimal accuracy. Finally, customize a prediction function to predict the emotion of the text by entering a piece of Chinese text.

# ２数据介绍及预处理工作
本实验所用数据包含两个维度，一是中文的文本内容text，主要来自微博上的对话内容。二是是根据对话内容所打的情感标签，主要包含６大类，分别是Happiness、Sorrow、Fear、Disgust、None和Love。而我们的任务就是挖掘出训练数据集中的语义特征并对测试集进行分类预测。原始数据如下：
![image](https://user-images.githubusercontent.com/65441161/143911565-84c52c4b-6612-4ed6-8bef-91f67dd5e025.png)

但是想要获得良好的预测结果，进行有效的数据预处理十分必要的。为此，我们做出以下工作：

１.首先我们对数据集查看缺失值，发现缺失值所占总体比例极小，故直接删除。

２.接下来我们进行去停用词。因为我们知道中文的对话文本内容中包含很多日常使用频率很高的常用词,如吧、吗、呢、啥等一些感叹词等。这些词和符号对系统分析预测文本的内容没有任何帮助,反而会增加计算的复杂度和增加系统开销,所以在使用这些文本数据之前必须要将它们清理干净。所以我们进行去停用词操作。

３.下一步进行分词操作。因为一段文本中往往有不止一个情感词，这些情感词对文本所要表达的情感起着非常关键的作用，有时候文本中的某一个情感词就决定了整段文本的情感，因此我们在进行情感分析之前就需要把这些情感词都找出来，那就必须先把文本分成词语的序列，也就是分词。对文本进行有效分词是进行情感分析之前一个非常重要的基础步骤。对于中文文本，句子与句子之间用句号隔开，但是句子内的词与词之间却没有明显的符号分隔，因此中文文本的分词难度要高于英文文本的分词。就目前来说，中文分词的方法主要有三种：基于理解的分词、基于统计的分词、基于字符串匹配的分词。目前比较成熟的分词服务主要有：阿里巴巴研发的阿里云NLP,腾讯研发的腾讯文智，复旦大学自然语言处理组研发的FudanNLP,哈工大社会计算与信息检索研究中心研发的LTP，以及在python语言中最流行的分词工具结巴分词等。其中结巴（jieba）分词完全开源，分词功能具有效率高、准确率高的特点，因此本文最终选择结巴（jieba）作为分词工具。

进行这些初步的数据预处理之后我们得到的数据如下图所示：
![image](https://user-images.githubusercontent.com/65441161/143913601-55f7b2ff-bce8-408c-889b-2c6f7e3824d8.png)



