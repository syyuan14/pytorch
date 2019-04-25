import torch
from torch.utils.data import DataLoader
from config import DefaultConfig as opt
from data import DogCat

def train(**kwargs):
    """
    训练
    """
    #根据命令行参数更新配置
    opt.parse(kwargs)
    #可视化
    #vis = Visualizer(opt.env)
    
    #step1 加载模型:
    model = getattr(models,opt.model)()
    if opt.load_model_path:
      model.load(opt.load_model_path)#加载训练好的参数
    if opt.use_gpu:model.cuda()

    #step2 数据：
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)

    train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)

    #step3:目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=opt.weight_decay)

    #统计指标：平滑处理之后的损失，还有混淆矩阵
    #######################待补充#########################
    
    #训练
    for epoch in range(opt.max_epoch):


      for ii,(data,label) in enumerate(train_dataloader):

        #训练模型
        if opt.use_gpu:
          input = input.cuda()
          target = input.cuda()

        optimizer.zero_grad()
        score = model(input)
        loss = criterion(score,target)
        loss.backward()
        optimizer.step()
        
        #更新统计指标及可视化

        if ii%opt.print_freq == opt.print_freq - 1:
          print('ii:{},loss:{}'.format(ii,loss))
    model.save()
    #计算在验证集上的指标及可视化
    

def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用于辅助训练
    """
    model.eval()
    for ii,data in enumerate(dataloader):
      input,label = data
      if opt.use_gpu:
        input = input.cuda()
        label = label.cuda()
      score = model(input)


def test(**kwargs):
    """
    测试(inference)
    """
    pass


def help():
    """
    打印帮助的信息
    """
    print('help')

if __name__ == '__main__':
  import fire
  fire.Fire()