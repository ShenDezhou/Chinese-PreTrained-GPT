# Chinese Pre-Trained GPT

本项目提供了面向中文的GPT预训练模型，旨在丰富中文自然语言处理资源，提供多元化的中文预训练模型选择。 我们欢迎各位专家学者下载使用，并共同促进和发展中文资源建设。

本项目基于Hugging Face官方GPT：https://github.com/openai/gpt-2

文档：https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/gpt2

# 新闻

**2023/5/27 发布GPT-12L-CN。**

历史新闻

# 简介

GPT是什么？  
```text
自然语言处理任务，例如问答、机器翻译、阅读理解和摘要，通常通过对特定任务的数据集进行监督学习来处理。
GPT作者证明，当在名为WebText的数百万个网页的新数据集上进行训练时，语言模型开始在没有任何显式监督的情况下学习这些任务。
当在文档加问题的条件下，语言模型生成的答案在CoQA数据集上达到55 F1，与使用127,000多个训练示例的4个基线系统中的3个的表现相当或更好。
语言模型的容量对于零样本任务转移的成功至关重要，并且增加容量会以对数线性方式改善跨任务的性能。
GPT作者的最大模型GPT-2是一个1.5B参数的Transformer，在零样本设置下在8个测试语言建模数据集中有7个达到了最先进的结果，但仍然无法很好地拟合WebText。
模型的样本反映了这些改进，包含有条理的文本段落。
这些发现表明，建立语言处理系统的有希望的路径是从自然发生的演示中学习执行任务。
```

# 模型下载

| 数据集   | owner      | model        | 语言    | 层数 | hidden | head | 参数量    |
|-------|------------|--------------|-------|----|--------|------|--------|
| 古诗[1] | Brian Shen | [gpt_12L_cn] | cn    | 12 | 768    | 12   | 102.0M |

[gpt_12L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/gpt/cn/pretrain/gpt2_L-12_H-768_A-12_CN.zip

> [1] 使用中文文本，使用BERT词表，按照GPT2网络结构训练而成。

# PyTorch/Tensorflow版本
提供PyTorch/Tensorflow1版本。

## 使用说明
Pytorch版本为：

```
GPT_L-12_H-768_A-12_CN.zip
    |- pytorch_model.bin     # 模型权重
    |- config.json           # 模型参数
    |- vocab.txt             # 分词词表
```

## 快速加载
依托于Huggingface-Transformers 3.1.0 ，可轻松调用以上模型。
```python
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
model = AutoModel.from_pretrained("MODEL_NAME")

或

tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = GPT2Model.from_pretrained("MODEL_NAME")

或

tokenizer = GPT2Tokenizer.from_pretrained("MODEL_NAME")
model = GPT2Model.from_pretrained("MODEL_NAME")
```

## 文本生成

以`GPT_L-12_H-768_A-12_CN`示例，如何进行文本生成。

```python
tokenizer = BertTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, return_dict=True, pad_token_id=0)

mask = pipeline('text-generation', model=model, tokenizer=tokenizer)
text = '明月几时有，'
response = mask(text, max_length=60)
```

本项目增加了一个colab工具，支持贪婪搜索、集束搜索、流畅度惩罚、采样等算法。
示例notebook见[text-generation_cn_gpt.ipynb](colab%2Ftext-generation_cn_gpt.ipynb)

## 引用
如果本目录中的内容对你的研究工作有所帮助，欢迎在论文中引用下述技术报告：


## 致谢
项目作者： Brian Shen. Twitter@dezhou.

建设该项目过程中参考了如下仓库，在这里表示感谢：
- GPT：https://github.com/openai/gpt-2


## 免责声明
本项目并非[GPT官方](https://github.com/openai/gpt-2) 发布的GPT模型。
该项目中的内容仅供技术研究参考，不作为任何结论性依据。
使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。


## 关注我们
欢迎关注知乎专栏号。
[深度学习兴趣小组](https://www.zhihu.com/column/thuil)


## 问题反馈 & 贡献
如有问题，请在GitHub Issue中提交。  
我们没有运营，鼓励网友互相帮助解决问题。  
如果发现实现上的问题或愿意共同建设该项目，请提交Pull Request。
