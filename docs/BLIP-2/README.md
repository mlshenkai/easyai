

## BLIP-2  Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

#### 论文地址： https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2301.12597.pdf

#### 代码地址： https://link.zhihu.com/?target=https%3A//github.com/salesforce/LAVIS/tree/main/projects/blip2

#### Demo地址： https://link.zhihu.com/?target=https%3A//huggingface.co/spaces/Salesforce/BLIP2

#### 知乎: https://zhuanlan.zhihu.com/p/698069132


### 1.1 背景和动机
视觉语言训练 (Vision-Language Pre-training, VLP) 的研究在过去几年中取得了快速的发展，研究者们开发了规模越来越大的预训练模型，不断推动各种下游任务的发展。但是，因为使用了大模型。大数据集，而且采取了端到端的训练，大多数最先进的视觉语言模型在预训练过程中会产生很高的计算代价和经济成本。

多模态的研究属于是视觉和语言研究领域的交叉，因此大家很自然地期望视觉和语言模型可以从现成的视觉，语言的预训练模型中获得。为了节约视觉语言模型在预训练过程的计算代价，本文提出的 BLIP-2 希望借助现成的预训练好的单模态视觉模型和单模态文本模型。

这样做的好处是：预训练的视觉模型能够提供高质量的视觉表征。预训练的语言模型，尤其是大型语言模型 (LLM)，提供了强大的语言生成和零样本迁移能力。为了减少计算成本并抵消灾难性遗忘的问题，单模态预训练模型在预训练期间保持冻结。

但是，简单的冻结预训练好的视觉模型的参数或者语言模型的参数会带来一个问题：就是视觉特征的空间和文本特征的空间不容易对齐。出现这个问题的原因是：文本模型 LLM 在单模态预训练期间没有看过对应的图片，视觉模型在单模态预训练期间没有看过对应的文本，所以这个对齐特别具有挑战性。

为了解决这个问题，BLIP-2 提出了一个轻量级的 Querying Transformer，如下图1所示。该 Transformer 分两个阶段进行预训练。Q-Former 是一个轻量级 Transformer，它使用一组可学习的 Query 向量从冻结的视觉编码器中提取视觉特征，并充当视觉编码器和文本编码器之间的瓶颈。Q-Former 把关键的视觉信息传递给 LLM，第一个预训练阶段，强制 Q-Former 学习与文本最相关的视觉表征。第二个预训练阶段，通过将 Q-Former 的输出连接到冻结的 LLM 来执行视觉语言生成学习，使其输出的视觉表征可以直接由 LLM 解释。这样一来，Q-Former 就可以有效地利用冻结的预训练图像模型和语言模型。
![图1：BLIP-2 模型。通过提出的 Q-Former 对齐视觉模态和文本模态](https://www.watchershen.cn:443/8zL8ad.png)