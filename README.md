个性化推荐算法实战

1 推荐算法综述
==============

1.1 个性化推荐算法综述
----------------------

![](media/de859246f6af788522cb6d3126d8a558.png)

![](media/fed8f66fe2a87ab43dce63e652ab95a5.png)

环境：python TensorFlow word2vec xgboost

高数微积分、线性代数、概率论相关知识

机器学习基本知识，数据挖掘大体了解

1.2 个性化召回算法综述
----------------------

### 1.2.1 什么是个性化召回

>   根据用户的属性行为上下文等信息从物品全集中选取其感兴趣的物品作为候选集

### 1.2.2 召回在推荐系统中的重要作用

召回决定了最终推荐结果的天花板

![](media/6f10f7753667f5dfd948e7f850afaa4f.png)

### 1.2.3 工业界个性化召回解析

分类

-   基于用户行为的

-   基于user profile的

-   基于隐语义的

工业界个性化召回架构

![](media/e1485a41b0e44c7b6ea0a9b69e9b139a.png)

2 个性化推荐召回算法
====================

2.1 LFM算法综述
---------------

### 2.1.1 个性化召回算法(latent factor model)综述

LFM算法的背景

<https://blog.csdn.net/bbbeoy/article/details/78646576>

应用于计算用户的toplike、计算item的topsim、计算item的topic

### 2.1.2 LFM理论知识与公式推导

LFM建模公式：

![](media/790c4468cd7f4a6138bf9cea0b60f792.png)

LFM loss function

![](media/be5e4dd081b423a8cf29d6ee7963ddf2.png)

LFM算法迭代

对上式求偏导：

![](media/ec631674e7e03792051c20603a829768.png)

使用梯度下降算法

![](media/8efce0b420da9edbe6520bb7cf4b8c46.png)

参数设定影响效果

-   负样本选取

-   隐特征F（10\~32）、正则参数、learning rate

### 2.1.3 LFM算法和CF算法的优缺点比较

理论基础

离线计算空间时间复杂度

在线推荐与推荐解释

3 personal rank算法
===================

3.1 personal rank算法的背景和物理意义
-------------------------------------

### 3.1.1 背景

用户行为很容易表示为图

图推荐在个性化推荐领域效果显著

### 3.1.2 二分图

二分图又称为二部图，是图论中的一种特殊模型。设G=(V,E)是一个无向图，如果顶点V可分割为两个互不相交的子集(A,B)，并且图中的每条边(i,j)所关联的两个顶点i和j分别属于这两个不同的顶点集(i
in A, j in B)，则称图G为一个二分图。

Example：

对userA来说，item_c和item_e哪个更值得推荐？

![](media/377fd507448f3edc58fc22ce90d62598.png)

### 3.1.3 物理意义

两个顶点之间连通路径数

两个顶点之间连通路径长度

两个顶点之间连通路径经过顶点的出度

Example:

分别由几条路径连通？

连通路径的长度分别是多少？

连通路径经过的顶点的出度分别是多少？

3.2 personal rank算法的数学公式推导
-----------------------------------

### 3.2.1 算法抽象-文字阐述

对用户A进行个性化推荐，从用户A结点开始在用户物品二分图random
walk，以alpha的概率从A的出边中等概率选择一条游走过去，到达该顶点后（举例顶点a），有alpha的概率继续从顶点a的出边中等概率选择一条继续游走到下一个结点，或者（1-alpha）的概率回到起点A，多次迭代。直到各顶点对于用户A的重要度收敛。

### 3.2.2 算法抽象-数学公式

![](media/5833308ce7f2a28ff0a383ead1a47cc2.png)

### 3.2.3 算法抽象-矩阵式

![](media/fa4ace6874d6f8b2ede42b8399d98a21.png)

4 item2vec算法
==============

4.1 个性化召回算法Item2vec背景与物理意义
----------------------------------------

### 4.1.1 背景

Item2item的推荐方式效果显著

NN model的特征抽象能力

算法论文：ITEM2VEC: NEURAL ITEM EMBEDDING FOR COLLABORATIVE
FILTERING(/doc/1603.4259.pdf)

https://blog.csdn.net/qq_35771020/article/details/89137392

### 4.1.2 物理意义

将用户的行为序列转化成item组成的句子

模仿word2vec训练word embedding将item embedding进行训练

### 4.1.3 缺陷

用户的行为序列时序性缺失

用户行为序列中的item强度是无区分性的

4.2 Item2vec算法应用主流程
--------------------------

从log中抽取用户行为序列

将行为序列当成语料训练word2vec得到item embedding

得到item sim关系用于推荐

Example:

![](media/344b61d2aa7005fc474326c875b67828.png)

4.3 Item2vec算法依赖model word2vec介绍
--------------------------------------

### 4.3.1 CBOW(continuous bag of words)

![](media/f0313790e883b524b7340165ebdf81a7.png)

### 4.3.2 Skip Gram

![](media/d13abe0088b8d339d5117dfc85edb9eb.png)

### 4.3.3 CBOW Word2vec数学原理

**问题抽象**：

![](media/3167f6aaa1b626a72417449f18a8d6ca.png)

Loss Function

![](media/88839c1b1d462e8fb73a7760a62dac45.png)

**梯度**：

![](media/160fa8356b49d16d0d94343ade0d41cd.png)

**CBOW训练主流程**

-   选取中心词w以及负采样NEG(w)

-   分别获得损失函数对xw与thetau的梯度

-   更新thetau以及中心词对应的context(w)的每一个词的词向量

### 4.3.4 Skip Gram Word2vec数学原理

**问题抽象**

![](media/762d6fb94b32cbf97df0ac4e106ba30b.png)

Loss Function

![](media/a3a65fe9eabfb8887dbc956fb4c58d86.png)

**Skip Gram word2vec训练主流程**

-   对于context(w)中任何一个词wc选取w的正负样本

-   计算Loss对theta以及对wc的偏导

-   更新wc对应的词向量

**负采样算法**

![](media/287a949a6b3c2a0fa6b07f6e72c9d505.png)

![](media/f02ab9775957bcbdd1b19c00a2f8a10b.png)

![](media/5151c079183d427e88c875cd712cc5f3.png)

5 content based算法
===================

5.1 个性化召回算法Content Based背景介绍
---------------------------------------

思路极简，可解释强

用户推荐的独立性

问世较早，流行度高

5.2 Content Based算法的主体流程介绍
-----------------------------------

### 5.2.1 Item Profile

Topic Finding

Genre Classify

### 5.2.2 User Profile

Genre/Topic

Time Decay

### 5.2.3 Online Recommendation

Find top K Genre/Topic

Get the best n item from fixed Genre/Topic

6 个性化召回算法总结与评估方法的介绍
====================================

6.1 个性化召回算法总结
----------------------

基于领域的

基于内容的

基于neural network的

![](media/211aa98183e3821981f980fed4f34fbe.png)

6.2 个性化召回算法评估
----------------------

离线评价准入

在线评价收益

### 6.2.1 Offline评价方法

评测新增算法推荐结果在测试集上的表现

![](media/5a511870308eea446e81b42fca3c9b01.png)

### 6.2.2 Online评价方法

定义指标

生产环境A/B test

7 排序综述
==========

7.1 什么是learn to Rank？
-------------------------

将个性化召回的物品候选集根据物品本身的属性结合用户的属性，上下文等信息给出展现优先级的过程

![](media/e358f8f7e0936f4805ccdc3ffba5055e.png)

7.2 排序在个性化推荐系统中的重要作用
------------------------------------

Rank决定了最终的推荐效果

![](media/4f5fc968ef91840c39e31cf27e78f9dc.png)

7.3 工业界推荐系统中排序架构解析
--------------------------------

单一的渐层模型

浅层模型的组合

深度学习模型

![](media/45b40348639e4f5acf8fe5a12d868897.png)

8 逻辑回归模型
==============

8.1 逻辑回归(logistic regression, LR)模型的背景知识介绍
-------------------------------------------------------

### 8.1.1 点击率预估与分类模型

<https://www.cnblogs.com/qcloud1001/p/7513982.html>

<https://blog.csdn.net/starzhou/article/details/51769561>

### 8.1.2 什么是LR

<https://blog.csdn.net/weixin_39445556/article/details/83930186>

### 8.1.3 Sigmoid函数

<https://www.jianshu.com/p/d4301dc529d9>

<https://blog.csdn.net/weixin_39445556/article/details/83930186>

Example

LR模型训练流程

-   从Log中获取训练样本与特征

-   Model参数学习

-   Model预测

### 8.1.4 LR Model优点与缺点

优点：易于理解，计算代价小

缺点：容易欠拟合，需要特征工程

8.2 逻辑回归模型的数学原理
--------------------------

### 8.2.1 阶跃函数及其导数

![](media/521d14691abb42a9c52fb2fe9a92dcc2.png)

### 8.2.2 LR Model Function

Model function

![](media/90b44097c1dd5c8963f0bda6fc3a59ef.png)

### 8.2.3 Loss Function

![](media/7a411e2c9f9690c80841d60e3f9617ac.png)

### 8.2.4 梯度

![](media/1ccec14efac8f4df934746c70cd5bc30.png)

![](media/3c4fe078ba8afa85a320318e49cdc07b.png)

梯度下降

![](media/4028ce04d0aa1793de81824f89670777.png)

### 8.2.5 正则化

什么是过拟合

<https://blog.csdn.net/qq_18254385/article/details/78428887>

L1正则化与L2正则化

<https://blog.csdn.net/zhaomengszu/article/details/81537197>

L1：

![](media/7cd027b06da36d3be6f7233b24a2038e.png)

L2:

![](media/e1270659bd5fddd22d3248f1826737ca.png)

8.3 样本选择与特征选择相关知识
------------------------------

### 8.3.1 Corpus

样本选择规则

样本过滤规则

### 8.3.2 Feature

特征的统计与分析

特征的选择

特征的预处理

![](media/53a90bc2afa9b1725c3fee49b2e3cc5f.png)

9 决策树算法
============

9.1 决策树背景知识介绍
----------------------

### 9.1.1 什么是决策树

决策树(Decision
Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。

<https://www.cnblogs.com/yonghao/p/5061873.html>

![](media/af8f54a6743966d4badf26c040ac9c14.png)

### 9.1.2 决策树构造原理

回归树的函数表示

![](media/1aea1b9af6e644094d4fd14f0851f860.png)

最优特征选取

![](media/6d7d4bdcec8dfd59f0ace721ee164511.png)

构建树的流程

-   遍历所有特征，特征的最佳划分对应的得分，选取最小得分的特征

-   将数据依据此选取的特征划分分成两部分

-   继续在左右两部分遍历变量找到划分特征直到满足停止条件

**CART生成**

<https://www.jianshu.com/p/b90a9ce05b28>

**回归树：平方误差最小化原则**

**分类树：基尼指数**

基尼指数

![](media/e805ccdf05558b9267f29c97a3a6ded7.png)

![](media/9be30f0ef9abde4678382d44ab1ea052.png)

9.2 梯度提升树的数学原理与构建方法
----------------------------------

### 9.2.1 什么是boosting

一种用来提高弱分类算法准确度的方法,这种方法通过构造一个预测函数系列,然后以一定的方式将他们组合成一个预测函数

如何改变训练数据的权重

如何组合多个基础model

### 9.2.2 Boosting Tree模型函数

![](media/cd9d5ee5be2447566e01fd9fc476963e.png)

![](media/3d45ae0f8bfe2e9a5ac907f5c8d811c4.png)

迭代损失函数

![](media/0ecb312b4961390aceda3a36cb2c2439.png)

### 9.2.3 提升树的算法流程

-   初始化$$f_{0}\left( x \right) = 0$$

-   对m=1,2,3,……计算残差rm，拟合rm，得到Tm

-   更新$$f_{m} = f_{m - 1} + T_{m}$$

Example

![](media/c15103c4bac742129941e76c606ad023.png)

![](media/faf26febde093d40a923b0ecca7fb55c.png)

### 9.2.4 梯度提升树

残差的数值改变

![](media/baa16bca8f5833be121e73fcefaebf87.png)

9.3 XGBoost数学原理与构建方法
-----------------------------

### 9.3.1 XGBoost模型函数

![](media/fd4f0b5daa33f79acba75502440e88b3.png)

### 9.3.2 优化目标的泰勒展开

![](media/af00c61e64363d57ab48a78b0b3c0422.png)

### 9.3.3 定义模型复杂度

![](media/e3b52ee5750dad9af62ce8e92853c01b.png)

### 9.3.4 目标转化

![](media/e4e13e048c924deeba012919b4318f10.png)

### 9.3.5 目标函数最优解

![](media/a3a406631f48d1617ad9897885be87d1.png)

### 9.3.6 最佳划分特征选取

![](media/1700eccdcc166f0b3e98ca91090eb5b0.png)

### 9.3.7 XGBoost总流程

-   初始化$$f_{0}\left( x \right) = 0$$

-   对m=1,2,3,……M应用选择最优划分特征的方法构造树

-   更新$$f_{m} = f_{m - 1} + learning\_ rate + T_{m}$$

9.4 gbdt与lr混合模型网络介绍
----------------------------

### 9.4.1 背景知识

Practical Lessons from Predicting Clicks on Ads at Facebook

逻辑回归需要繁琐的特征处理

树模型的feature transform能力

### 9.4.2 模型网络

![](media/903e13483b926105f5f86213ad936af5.png)

### 9.4.3 优缺点总结

优点：利用树模型做特征转化

缺点：两个模型单独训练不是联合训练

10 深度学习
===========

10.1 深度学习背景介绍
---------------------

### 10.1.1 神经元

![](media/0e97e5e2e1620edce5ab577fc399d229.png)

### 10.1.2 激活函数

sigmoid

tanh

relu

### 10.1.3 神经网络

![](media/c01a0a85b18cae6ac7985873a8dcbf17.png)

### 10.1.4 DL & ML difference

![](media/00645f0c88f98cade32df52435432c26.png)

10.2 DNN网络结构与数学原理
--------------------------

### 10.2.1 DNN网络结构

![](media/854a52bbb726637799da7e47c5574ea9.png)

### 10.2.2 DNN模型参数

隐层层数，每个隐层神经元个数，激活函数

输入输出层向量维度

不同层之间神经元的连接权重W和偏移值B

10.2.3 前向传播

节点的输出值

![](media/d17d1320a29be6ae1c8f860d5b524856.png)

![](media/f214026916c810a114e191935ac625bd.png)

![](media/6dd780807378922fd26d261095d49c95.png)

### 10.2.3 反向传播

![](media/b30664131af598997798bea84c14c17f.png)

核心

![](media/fea431859b74021badecc2cb2facb9a4.png)

-   对输入x，设置合理的输入向量

-   前向传播逐层逐个神经元求解加权和与激活值

-   对于输出层求解输出层损失函数对于z值的偏导

-   反向传播逐层求解损失函数对z值的偏导

-   得到w与b的梯度

10.3 WD(wide and deep)网络结构与数学原理
----------------------------------------

### 10.3.1 w&d的物理意义

论文：wide & deep learning for Recommender Systems

Generalization and Memorization(/doc/p7-cheng.pdf)

### 10.3.2 w&d的网络结构

![](media/c9d3097dcff696eee654f0d674f6b5ae.png)

x

模型输出

![](media/9ecd47f97b1b955ca6d56bb13a9c3ae5.png)

### 10.3.3 WD model的反向传播

Wide侧参数学习

![](media/1299b6c4913285a42f891df4bf20e5dc.png)

Deep侧参数学习

![](media/a1e5b01ddc8e9af997dfe8f726ffe0ad.png)

### 10.3.4 server架构

![](media/52118f60b1c8300373944b3fd1eb6584.png)

11 学习排序部分总结
===================

11.1 效果回顾
-------------

![](media/6f1a84b2b4541806fa7bfb53c0df6e86.png)

11.2 离线评估
-------------

Model cv

Model test and data performance

11.3 在线评估
-------------

业务指标

平均点击位置

11.4 特征维度浅析
-----------------

特征维度

User 、Item、Context、UI Relation、Statics Supplement

特征数目

11.5 Rank技术展望
-----------------

多目标学习

强化学习

12 总结
=======

12.1 个性化推荐算法的离线架构
-----------------------------

![](media/e6baea597567582e37e3ace7a023f36c.png)

12.2 个性化推荐算法的在Recall线架构
-----------------------------------

![](media/7d355acc0554a9466db66fbb80729c57.png)

12.3 个性化推荐算法的Rank在线架构
---------------------------------

![](media/167d8a9c4981f5a4d7df3d502cff29ea.png)

12.4 个性化推荐算法的回顾
-------------------------

Recall：CF, LFM, Personal Rank, Item2vec, ContentBased

Rank: LR, GBDT, LR + GBDT, WD
