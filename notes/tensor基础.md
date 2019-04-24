1. Tensor的创建
  import torch
  1.1 torch.Tensor(size)
      tensor_a = torch.Tensor(2,3)#括号内直接输入形状然后随机生成
      tensor_b = torch.Tensor([2,3])#括号内输入要生成的张量表达式进行生成
  1.2 torch.tensor(data,)
      tensor_a = torch.Tensor([2,3])#与torch.Tensor的第二种用法一样,括号内直接放所要生的数据而不是形状
  1.3 torch.ones(*sizes)
      tensor_one = torch.ones([2,3])#生成全一的两行三列张量
      tensor_one = torch.ones(2,3)#生成两行三列的全一张量
  1.4 torch.zeros(*sizes)
      tensor_zeros = torch.zeros(2,3)#生成两行三列的全零张量
      tensor_zeros = torch.zeros([2,3])#生成两行三列的全零张量
  1.5 torch.eye(*sizes)
      tensor_a = torch.eye(2,2)#生成两行两列的单位张量
  1.6 torch.arange(s,e,step)#从s到e,步长为step
      tensor_a = torch.arange(1,9,2)#生成的张量是一维的,即tensor([1,3,5,7])
  1.7 torch.linspace(s,e,steps)#从s到e均匀分成steps份
      tensor_a = torch.linspace(0,9,4)#生成的张量是一维的,即tensor([0,3,6,9])
  1.8 torch.rand/randn(*sizes)#均匀/正态分布
      tensor_a = torch.rand(2,3)
  1.9 torch.normal(mean,std)/uniform(from,to)#正态分布/均匀分布
  2.0 torch.randperm(m)#随机排列
2. Tensor的基本操作：
  b = torch.Tensor(2,3)
  a = b.tolist()#a为张量b转化为的list
  b.size()/b.shape#返回b的形状
  b.numel()#b中元素总个数
  b.view(1,6)#把b转化为1行6列改变形状
  b.unsqueeze()#填充维数
  #例如：
  b.size()= torch.Size([2,2,2])
  b.unsqueeze(0).size() = torch.Size([1,2,2,2])
  b.unsqueeze(2).size() = torch.Size([2,2,1,2])
  #例如
  b.squeeze()#压缩维数(只能压缩维度为一的维数)
  b.size() = torch.Size([1,2,3,1])
  b.squeeze().size() = torch.Size([2,3])#不传参数压缩所有维度为一的维数
  b.squeeze(0).size() = torch.Size([2,3,1])
  #resize是另一种可用来调整size的方法，但与view不同，他可以修改tensor的大小，如果新大小超过了原大小，会自动分配内存空间，
  #而如果新大小小于原大小，则之间的数据依旧会被保存例如：
  b.size() = torch.Size([2,3])
  b.resize_(3,3)#多出的行会用随机数据进行填充
  b.resize_(1,3)#少的数据系统仍然会保存
3. Tensor的索引
4. 一个简单的线性回归问题
  1. 拿到训练数据
  2. 建立模型
  3. 拿到模型的预测值
  4. 计算loss
  5. 反向传递优化参数
  
