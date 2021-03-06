# Chinese_text_emotion_analysis_based_on_LSTM
# 基于LSTM网络的中文文本情感分析任务
# １　摘要
　　本项目基于LSTM（长短期记忆网络）来实现对中文文本情感的分类及预测。本实验采用高质量的六情感微博数据，利用Jieba进行分词，将中文文本序列化，同时进行词嵌入，再使用LSTM算法进行核心的分类器训练，调整模型实现最优的准确率。最后自定义一个预测函数，通过输入一段中文文本来预测文本的情感。

　　This project is based on LSTM (Short- and Short-Term Memory Network) to classify and predict Chinese text emotions. This experiment uses high-quality six-emotion micro-blog data, uses Jieba to serialize Chinese text, embeds words, and then uses LSTM algorithm to train the core classifier, adjust the model to achieve optimal accuracy. Finally, customize a prediction function to predict the emotion of the text by entering a piece of Chinese text.

# ２　数据介绍及预处理工作
　　本实验所用数据包含两个维度，一是中文的文本内容text，主要来自微博上的对话内容。二是是根据对话内容所打的情感标签，主要包含６大类，分别是Happiness、Sorrow、Fear、Disgust、None和Love。而我们的任务就是挖掘出训练数据集中的语义特征并对测试集进行分类预测。

原始数据如下：

![image](https://user-images.githubusercontent.com/65441161/143911565-84c52c4b-6612-4ed6-8bef-91f67dd5e025.png)

　　但是想要获得良好的预测结果，进行有效的数据预处理十分必要的。为此，我们做出以下工作：

# ２.１　缺失值处理
　　先我们对数据集查看缺失值，发现缺失值所占总体比例极小，故直接删除。

# ２.２　去停用词和分词处理
　　接下来我们进行去停用词。因为我们知道中文的对话文本内容中包含很多日常使用频率很高的常用词,如吧、吗、呢、啥等一些感叹词等。这些词和符号对系统分析预测文本的内容没有任何帮助,反而会增加计算的复杂度和增加系统开销,所以在使用这些文本数据之前必须要将它们清理干净。所以我们进行去停用词操作。

　　下一步进行分词操作。因为一段文本中往往有不止一个情感词，这些情感词对文本所要表达的情感起着非常关键的作用，有时候文本中的某一个情感词就决定了整段文本的情感，因此我们在进行情感分析之前就需要把这些情感词都找出来，那就必须先把文本分成词语的序列，也就是分词。对文本进行有效分词是进行情感分析之前一个非常重要的基础步骤。对于中文文本，句子与句子之间用句号隔开，但是句子内的词与词之间却没有明显的符号分隔，因此中文文本的分词难度要高于英文文本的分词。就目前来说，中文分词的方法主要有三种：基于理解的分词、基于统计的分词、基于字符串匹配的分词。目前比较成熟的分词服务主要有：阿里巴巴研发的阿里云NLP,腾讯研发的腾讯文智，复旦大学自然语言处理组研发的FudanNLP,哈工大社会计算与信息检索研究中心研发的LTP，以及在python语言中最流行的分词工具结巴分词等。其中结巴（jieba）分词完全开源，分词功能具有效率高、准确率高的特点，因此本文最终选择结巴（jieba）作为分词工具。

　　进行这些初步的数据预处理之后我们得到的数据如下图所示：
  
![image](https://user-images.githubusercontent.com/65441161/143913601-55f7b2ff-bce8-408c-889b-2c6f7e3824d8.png)

# ２.３　序列化和词嵌入
　　但是我们并不能直接把中文文本带入模型，所以我们要把中文文本进行序列化操作和词嵌入操作。

　　我们要将text数据进行向量化处理,我们要将每条text转换成一个整数序列的向量。本文选择keras的分词器（Tokenizer）进行序列化。使用Tokenizer根据数据集生成词典，并且将词典保存为.json文件，以便预测的时候可以再次使用该词典。如下图所示，词典中每个词存在且仅存在一个索引与之对应，并且Tokenizer还会统计数据集中各个词出现的频数。然后通过texts_to_sequences（）方法将文本序列化，也就是将文本转换成由索引构成的序列。
  
![image](https://user-images.githubusercontent.com/65441161/143914764-980a97a8-b20d-4fc7-96b6-12b96cd5be11.png)

　　接下来的操作就是词嵌入。因为要想让计算机理解文字的含义，就必须用某种方法将计算机无法理解的文字内容转化为计算机可以理解的数字、矩阵向量、符号，使其在计算中获取信息。为了解决此问题，我们引入词嵌入，词嵌入的本质上是一个映射，它可以将词语或者短语映射到一个高维度的空间中形成词向量，如果有两个词的词义是相近的或者相同的，那么这两个词在映射后他们在空间中的距离也是接近的。目前比较流行的词嵌入主要有：谷歌在新闻数据集上训练出来的word2vec、斯坦福大学提出并且训练的GloVe。中文领域的词嵌入相对较少。在国内，腾讯人工智能实验室公布了特定于中文领域的词嵌入映射文件Tencent_AILab_ChineseEmbedding.txt，本文件将中文词汇分别映射到多维空间中，本词嵌入文件不论在词汇覆盖率还是在映射后的效果都很高，因此本文选择Tencent_AILab_ChineseEmbedding.txt进行词嵌入。由于需要对这么多个词中的每一个都映射为多维的向量并存储到文件中，这必然导致最后的文件占用的内存非常的大，解压出来的腾讯词嵌入文件大小达16.0G，而本计算机的内存只有8G，显然无法直接将文件载入内存中。为了解决此问题，本文专门创建了词嵌入文件的映射文件：embeddings_map_index.txt，该文件存储的是原词嵌入文件的索引，也就是只保留原词嵌入文件中的词汇字段、词汇在词嵌入文件中的起始位置、词汇在词嵌入文件中占用的长度。只要根据本映射文件就可以快速的从原词嵌入文件中读取需要的词向量，而不需将整个词嵌入文件读入内存。（注：由于Tencent_AILab_ChineseEmbedding.txt和embeddings_map_index.txt两个文件太大，故未上传，有需要可以在腾讯人工智能实验室官网下载使用。）
  
# ２.４　文本填充  
　　我们可知道将数据输入网络之前需要先对数据序列进行均匀地切分。由于文本的长度不一，因此不同文本序列化后的向量长度也是不同的，这样就不能进行统均匀地切分。为了解决这一问题，我们需要进行文本填充，先指定填充长度MAXLEN，若序列化后的文本向量长度小于MAXLEN，那么就用“0”进行填充，直到长度文本向量长度刚好为MAXLEN。若序列化后的文本向量长度大于MAXLEN，那么就将超出的部分截断，只能保留长度为MAXLEN的文本向量。

# ３　模型的搭建与训练
# ３.１　模型搭建  
　　训练和测试的数据集都准备好以后,接下来我们要定义一个LSTM的序列模型:

　　模型的第一次是嵌入层(Embedding)，它使用长度为100的向量来表示每一个词语。SpatialDropout1D层在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。LSTM层包含100个记忆单元。输出层为包含６个分类的全连接层。由于是多分类，所以激活函数设置为'softmax'，损失函数为分类交叉熵categorical_crossentropy。
  
　　可以看到定义好之后如下图所示：
  
  ![image](https://user-images.githubusercontent.com/65441161/143916671-437c6794-149b-4c76-bc1b-905d962e2175.png)
  
  # ３.２　模型训练
　搭建好模型之后我们就需要对模型进行训练，开始我们设置３个训练周期，并定义好一些超参数（后边调优时会对参数进行微调）。同时我们可以画出模型的准确率和ｌｏｓｓ函数变化曲线，如下图所示：
 
  ![image](https://user-images.githubusercontent.com/65441161/143917367-4606333e-e77c-4e96-bb96-a1b5d77ed16a.png)
![image](https://user-images.githubusercontent.com/65441161/144067117-344fa0d8-874c-4bf2-9bf5-b0777224c543.png)![image](https://user-images.githubusercontent.com/65441161/144067186-81e746f7-7195-4a03-9e6e-dea1ac440eb1.png)

　 从上图中我们可以看见,随着训练周期的增加,模型在训练集中损失越来越。准确率越来越高；而在测试集中, 损失随着训练周期的增加由一开始的从大逐步变小,再逐步变大。准确率随着训练周期的增加由一开始的从小逐步变大，再逐步变小。
 # ３.３　模型评估
　　多分类模型一般不使用准确率(accuracy)来评估模型的质量,因为accuracy不能反应出每一个分类的准确性,因为当训练数据不平衡(有的类数据很多,有的类数据很少)时，accuracy不能反映出模型的实际预测精度,这时候我们就需要借助于F1分数、ROC等指标来评估模型。而本文正是采用F1分数作为评估标准，评估结果如下图：
  
  ![image](https://user-images.githubusercontent.com/65441161/143918460-529a9242-003d-49e2-9859-e121efd956b7.png)

　　从以上F1分数上看,各类预测并不是很均衡，"None"类的F1分数最大(７４%)，“Fear”类F1分数却出奇的差，我的想法是可能是因为“Fear”分类的训练数据最少只有２０多条,使得模型学习的不够充分,导致预测失误较多。当然，对于参数的微调也是有待进一步提高。
# ４　自定义模型预测
最后我们自定义了一个预测函数，通过输入一段中文文本内容就可以实现对于文本情绪的预测，示例如下：

![image](https://user-images.githubusercontent.com/65441161/143919067-4f44368a-c0ba-4267-bc74-302d5a217197.png)

