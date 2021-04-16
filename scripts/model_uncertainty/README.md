

Domain adaptation (DA) ：有目标domain的无标签数据或少量标签数据，可继续训练

Domain generalization (DG)：无

DG is a harder problem than DA in that it
makes fewer assumptions (target data not required) but for
the same reasons, it may be more valuable if solved.



# Domain Generalization

The majority of proposed deep learning methods for domain generalization fall into one of two categories:

* 1) Learning a single **domain invariant representation** do not explicitly regularize z_y using d. 
* 2) **Ensembling models**, each trained on an individual domain from the training set. The size of models in this category scales linearly with the amount of training domains. This leads to slow inference if the number of training domains is large.

Follow 第一点，学习一个**domain invariant representation**。



## 难点：

* 1.多数方法使用了domain标签，而驾驶数据中可能没有合理的domain标签
  * 没有domain标签如何学习domain invariant representation ？？？

* 2.多数方法是分类问题，而轨迹预测是生成问题，比分类问题维度更高
  * 利用无监督GAN得到decoder





