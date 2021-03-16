from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.parameter import Parameter
import argparse
torch.manual_seed(0)
np.random.seed(1234)

# DPSH(IJCAI2016)
# paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)
# code [DPSH-pytorch](https://github.com/jiangqy/DPSH-pytorch)

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu", default="0", type=int, help="whith gpu to use")
    args = parser.parse_args()
    
    return args


def mk_save_dir(types, root_path, settings):
    output_dir = "./%s/%s"%(types, root_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "%s"%("__".join([str(i) for i in settings]))
    return output_file


def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        
        "opt_type": optim.RMSprop,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "info": "[DPSH_D]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
#         "dataset": "cifar_10",
#         "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 200,
        "test_map": 5,
        "save_path": "save/DPSH_D",
        # "device":torch.device("cpu"),
        "class_bit": 12,
        "bit_list": [48],
        
    }
    config = config_dataset(config)
    return config


class DPSH_D_Loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSH_D_Loss, self).__init__()
        self.class_bit = config["class_bit"]
        self.U = torch.zeros(config["num_train"], config["n_class"], self.class_bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        self.w_D = Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(config["n_class"], bit, self.class_bit)), requires_grad=True)
    
    
    def forward(self, u, y, ind, config): 
        n_train = config["num_train"]
        n_bs = y.size()[0]
        n_cls = config["n_class"]
#         self.Y_ = self.Y.unsqueeze(0).expand(n_bs,-1,-1)
        
        self.Y[ind, :] = y.float()
        y_ = y.unsqueeze(1).expand(-1, n_train, -1)
        self.Y_ = self.Y.unsqueeze(0).expand(n_bs,-1, -1)
        s = self.Y_ * y_ # get label-wise similarity: bs, n_train, n_class
        u_ = torch.matmul(u, self.w_D.to(config["device"])).permute(1,0,2)
        self.U[ind, :, :] = u_.data
        inner_product = torch.sum(self.U.unsqueeze(0).expand(n_bs,-1,-1,-1) * u_.unsqueeze(1).expand(-1,n_train,  -1, -1), dim=-1) #bs, n_train, n_class
#         
#         inner_product = 
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product # bs, train_num
        likelihood_loss = likelihood_loss.mean()

        quantization_loss = config["alpha"] * (u_ - u_.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss


    
def train_val(config, bit):
    config["device"] = torch.device("cuda:%d"%config['gpu'] if torch.cuda.is_available() else "cpu")
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)
    criterion = DPSH_D_Loss(config, bit).to(device)
    optimizer = optim.RMSprop(list(net.parameters()) + list(criterion.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    
    settings = [bit, config["class_bit"], config["lr"]]
    performance_file = mk_save_dir("performance", "%s/DPSH_D/"%config["dataset"] , settings)
    
    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = comput_ClassAware_result(test_loader, net, criterion.w_D.to(config["device"]), device=device) # tst_binary: n_class, n_sample, n_bit
            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = comput_ClassAware_result(dataset_loader, net, criterion.w_D.to(config["device"]), device=device)
            # print("calculating map.......")
#             mAP = CalcTopMap_ClassAware_gpu(trn_binary, tst_binary, trn_label, tst_label, config["topK"])
            mAP = CalcTopMap_ClassAware(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),config["topK"])
            
            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-model.pt"))
            performance_log = "%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP)
            print(performance_log)
            output_f = open(performance_file, "a")
            output_f.write(performance_log + "\n")
            output_f.close()
            print(config)


if __name__ == "__main__":
    config = get_config()
    paras = get_cmd()
    for k, v in paras.__dict__.items():
        config[k] = v
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
