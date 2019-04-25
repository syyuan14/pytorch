import torch
import time
class BasicModule(torch.nn.Module):
  """
  封装了nn.Module,主要提供save和load两个方法
  """
  def __init__(self):
    super(BasicModule,self).__init__()
    self.model_name = str(type(self))#模型的默认名称
  def load(self,path):
    """
    可加载指定路径的模型
    """
    self.load_state_dict(torch.load(path))
  def save(self,name=None):
    """
    保存模型,默认使用“模型名称+时间”作为文件名
    如AlexNet_0710_23:57:29.pth
    """
    if name is None:
      prefix = 'checkpoints/' + self.model_name + '_'
      name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
    torch.save(self.state_dict(),name)
    return name

class Flat(torch.nn.Module):
  """
  把输入reshape成(batch_size,dim_lenght)
  """
  def __init__(self):
    super(Flat,self).__init__()
  def forward(self,x):
    return x.view(x.size(0),-1)