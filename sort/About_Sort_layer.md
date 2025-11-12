# 一些经验
- 关于每个特征的embedding的大小的选择，个人通过试验得到，首先将user_id和movie_id的embedding size设置为16，其余的设置为8。然后每次将所有的embedding size都减半，发现模型性能有所上升，这可能源于embedding size较大时模型出现过拟合。最终得到的结果是user_id和movie_id的embedding size设置为4，其余的设置为2。
## Deep
- 关于deep模型MLP的维度，个人认为应当和输入的特征concat后的维度相近，这样模型的表达能力会更强一些。当然，首先要把每个特征的embedding size设置到合适的大小。
## DCN
- DCN的加入增加了特征的交叉能力，通过实验发现，DCN层的输出维度应该和Deep Model的输出维度相近，这样在concat时不会出现某一部分占据主导地位的情况，从而提升模型的效果。之前DCN的输出维度过大，导致模型效果不佳，低于Deep。