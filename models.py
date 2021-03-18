from utils.tools import *
from network import *
import torch
import torch.optim as optim
import time
import numpy as np
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.parameter import Parameter
torch.manual_seed(0)
np.random.seed(1234)


class DPSH_D(torch.nn.Module):
# DPSH(IJCAI2016)
# paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)
# code [DPSH-pytorch](https://github.com/jiangqy/DPSH-pytorch)
    def __init__(self, conf):
        super(DPSH_D, self).__init__()
        self.hash_bit = conf["hash_bit"]
        self.class_bit = conf["class_bit"]
        self.n_class = conf["n_class"]
        self.conf = conf
        if conf["setting"] == 1:
            self.backbone_net = AlexNet(self.hash_bit)
            self.w_D = Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(conf["n_class"], self.hash_bit, self.class_bit)), requires_grad=self.conf["wD_train"])
        elif conf["setting"] == 2:
            self.backbone_net = AlexNet_class_wise(self.class_bit, self.n_class)
        if conf["class_wise"]:
            self.U = torch.zeros(conf["num_train"], conf["n_class"], self.class_bit).float().to(self.conf["device"])
        else:
            self.U = torch.zeros(config["num_train"], self.hash_bit).float().to(config["device"])
        self.Y = torch.zeros(conf["num_train"], conf["n_class"]).float().to(self.conf["device"])
        
        
    def get_loss(self, u, y, ind):
        n_train = self.conf["num_train"]
        n_batch = y.size()[0]
        n_cls = self.conf["n_class"]
        self.Y[ind, :] = y.float()
        y_ = y.unsqueeze(1).expand(-1, n_train, -1)
        self.Y_ = self.Y.unsqueeze(0).expand(n_batch,-1, -1)
        s = self.Y_ * y_ # get label-wise similarity: bs, n_train, n_class
        
        
        if not self.conf["class_wise"]:
            s = torch.max(s,dim=-1)[0]
            self.U[ind, :] = u.data
            inner_product = u @ self.U.t() * 0.5
            likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product # bs, train_num
            
        else:
            if self.conf["setting"] == 1:
                u_ = torch.matmul(u, self.w_D).permute(1,0,2)
            elif self.conf["setting"] == 2:
                u_ = torch.reshape(u, [n_batch, self.n_class, self.class_bit])
        
            self.U[ind, :, :] = u_.data
            
            inner_product = torch.sum(self.U.unsqueeze(0).expand(n_batch,-1,-1,-1) * u_.unsqueeze(1).expand(-1, n_train,  -1, -1), dim=-1) * 0.5 #bs, n_train, n_class
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product # bs, train_num, cls
        if self.conf["class_agg"] == "ave":
            likelihood_loss = torch.mean(likelihood_loss, dim=-1)
        elif self.conf["class_agg"] == "max":
            likelihood_loss = torch.max(likelihood_loss, dim=-1)[0]
        elif self.conf["class_agg"] == "weight":
            class_info = torch.mean(torch.mean(s, dim=0), dim=0)/torch.max(torch.mean(torch.mean(s, dim=0), dim=0))
            class_mask = class_info > 0
            
            likelihood_loss = torch.mean(class_mask * likelihood_loss/(class_info+1e-9), dim=-1)
        likelihood_loss = likelihood_loss.mean()
        return likelihood_loss
        
        
    def forward(self, image, label, ind):
        u = self.backbone_net(image)
        likelihood_loss = self.get_loss(u, label, ind)
        quantization_loss = self.conf["alpha"] * (u - u.sign()).pow(2).mean()
        return likelihood_loss + quantization_loss
    
    

class DPSH(torch.nn.Module):
# DPSH(IJCAI2016)
# paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)
# code [DPSH-pytorch](https://github.com/jiangqy/DPSH-pytorch)
    def __init__(self, conf):
        super(DPSH, self).__init__()
        self.hash_bit = conf["hash_bit"]
        self.class_bit = conf["class_bit"]
        self.n_class = conf["n_class"]
        self.conf = conf
        self.backbone_net = AlexNet(self.hash_bit)
        self.U = torch.zeros(conf["num_train"], self.hash_bit).float().to(conf["device"])
        self.Y = torch.zeros(conf["num_train"], conf["n_class"]).float().to(conf["device"])
        
        
    def get_loss(self, u, y, ind):
        n_train = self.conf["num_train"]
        n_batch = y.size()[0]
        n_cls = self.conf["n_class"]
        self.Y[ind, :] = y.float()
        y_ = y.unsqueeze(1).expand(-1, n_train, -1)
        self.Y_ = self.Y.unsqueeze(0).expand(n_batch,-1, -1)
        s = self.Y_ * y_ # get label-wise similarity: bs, n_train, n_class

        s = torch.max(s,dim=-1)[0]
        self.U[ind, :] = u.data
        inner_product = u @ self.U.t() * 0.5
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product # bs, train_num

        likelihood_loss = likelihood_loss.mean()
        return likelihood_loss
        
        
    def forward(self, image, label, ind):
        u = self.backbone_net(image)
        likelihood_loss = self.get_loss(u, label, ind)
        quantization_loss = self.conf["alpha"] * (u - u.sign()).pow(2).mean()
        return likelihood_loss + quantization_loss
    


