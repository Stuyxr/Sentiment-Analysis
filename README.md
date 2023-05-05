### 中文词向量模型

<https://github.com/Embedding/Chinese-Word-Vectors> 

sgns.weibo.bigram

### 运行环境

pytorch1.0.1

### 训练模型

```
python load_data.py
python sentiment_analysis.py
```

### 测试模型

```
python test.py
```

### 说明

“train_data”目录下提供10000条训练数据，包括5000条积极情感文本（sample.positive.txt）和5000条消极情感文本（sample.negative.txt）；

文件为“UTF-8”编码，数据以xml格式存储，格式如下：

```
<review id="n">
xxx
</review>
```

每个“review”标签是一条训练数据，“id”是训练数据编号（0到9999），标签内容“xxx”为文本内容。
“test_data”目录下是文件“test.txt”，包含2500条未知类别（积极或消极）的测试数据，使用学习的系统对其进行预测。
文件为“UTF-8”编码，数据以xml格式存储，格式如下：

```
<review id="n">
xxx
</review>
```

每个“review”标签是一条测试数据，“id”是测试数据编号（0到2499），标签内容“xxx”为文本内容。
对测试数据进行预测，积极用“1”表示；消极用“0”表示。
