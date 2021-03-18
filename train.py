from utils.tools import *
from models import *
import yaml
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
    parser.add_argument("-gpu", "--gpu", default=0, type=int, help="whith gpu to use")
    parser.add_argument("-m", "--model", default="DPSH_D", type=str, help="which model to train")
    parser.add_argument("-d", "--dataset", default="nuswide_21", type=str, help="which dataset to train")
    parser.add_argument("-st", "--setting", default=1, type=int, help="1: AlexNet, 2: AlexNet_class_wise")
    parser.add_argument("-lr", "--lr", default=0.00001, type=float, help="learning rate for train")
    parser.add_argument("-wd", "--weight_decay", default=0.00001, type=float, help="weight decay for train")
    parser.add_argument("-hb", "--hash_bit", default=48, type=int, help="length of hash code")
    parser.add_argument("-cb", "--class_bit", default=48, type=int, help="length of hash code/class")
    parser.add_argument("-cw", "--class_wise", default=1, type=int, help="if apply class wise, if set to 0, it equals to DPSH")
    parser.add_argument("-dl", "--diff_lr", default=0, type=int, help="if use different lr for net and wD")
    parser.add_argument("-ca", "--class_agg", default="ave", type=str, help="how to aggregate the logloss of different classes: ave, max")
    args = parser.parse_args()
    return args



def mk_save_dir(types, root_path, settings):
    output_dir = "./%s/%s"%(types, root_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "%s"%("_".join([str(i) for i in settings]))
    return output_file

    
    
def train_eval(conf):
    conf["device"] = torch.device("cuda:%d"%conf['gpu'] if torch.cuda.is_available() else "cpu")
    device = conf["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(conf)
    conf["num_train"] = num_train
    if conf["model"] == "DPSH_D":
        model = DPSH_D(conf)
    elif conf["model"] == "DPSH":
        model = DPSH(conf)
    model.to(device=conf["device"])
        
    if conf["diff_lr"]:
        optimizer = optim.Adam([{"params":model.backbone_net.parameters(), "lr":conf["lr"], "weight_decay":conf["weight_decay"]},
                              {"params":model.w_D, "lr":conf["lr"]*10, "weight_decay":conf["weight_decay"]}])
        
    else:
        optimizer = optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
        
    for name, param in model.named_parameters():
        print(name)
        
    settings = ["bit", str(conf["hash_bit"]),"lr", str(conf["lr"]), "wd", str(conf["weight_decay"])]
    if conf["model"] == "DPSH_D":
        if conf["class_wise"]:
            settings.append("class_bit_%s"%str(conf["class_bit"])) 
            if conf["setting"] == 1:
                settings.append("type1")
                if conf["wD_train"]:
                    settings.append("train_wD")
        if conf["diff_lr"]: 
            settings.append("diff_lr")
    performance_file = mk_save_dir("performance", "%s/%s/"%(conf["dataset"],conf["model"]) , settings)
    result_path = "%s/%s/%s/"%(conf["result_path"], conf["dataset"], conf["model"])
    model_save_path = "%s/%s/%s/"%(conf["model_path"], conf["dataset"], conf["model"])
    Best_mAP = 0
    Best_epoch = 0
    
    for epoch in range(conf["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] %s, dataset:%s, training...." % (
            conf["model"], epoch + 1, conf["epoch"], current_time, "_".join(settings), conf["dataset"]), end="")

        model.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss = model(image, label, ind)
            train_loss += loss.detach().cpu()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        
        if (epoch + 1) % conf["test_interval"] == 0:
            model.eval()
            if conf["class_wise"] and conf["model"] == "DPSH_D":
                tst_binary, tst_label = compute_ClassAware_result(test_loader, model.backbone_net, model.w_D, device=device) # tst_binary: n_class, n_sample, n_bit
                trn_binary, trn_label = compute_ClassAware_result(dataset_loader, model.backbone_net, model.w_D, device=device)

                mAP = CalcTopMap_ClassAware(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),conf["topK"])
                
            else:
                tst_binary, tst_label = compute_result(test_loader, model.backbone_net, device=device)
                trn_binary, trn_label = compute_result(dataset_loader, model.backbone_net, device=device)
                mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), conf["topK"])
            
            if mAP > Best_mAP:
                Best_mAP = mAP
                
                if conf["save_flag"]:
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        
                    print("save in ", conf["model_path"])
                    old_model_name = model_save_path + "%s-model.pt"%Best_epoch
                    new_model_name = model_save_path + "%s-model.pt"%epoch
                    torch.save(model.backbone_net.state_dict(),new_model_name)
                    if Best_epoch != 0:
                        os.remove(old_model_name)
                    

                    np.save(result_path + "trn_binary_%d.npy"%conf["hash_bit"], trn_binary.numpy())
                    np.save(result_path + "tst_binary_%d.npy"%conf["hash_bit"], tst_binary.numpy())
                    np.save(result_path + "trn_label_%d.npy"%conf["hash_bit"], trn_label.numpy())
                    np.save(result_path + "tst_label_%d.npy"%conf["hash_bit"], tst_label.numpy())
                    
                Best_epoch = epoch 
                    
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            performance_log = "epoch:%d, %s, MAP:%.3f, Best MAP: %.3f (%d)" % (
                epoch + 1, " ".join(settings), mAP, Best_mAP, Best_epoch)
            print("%s dataset: %s, %s"%(current_time, conf["dataset"], performance_log))
            output_f = open(performance_file, "a")
            output_f.write(performance_log + "\n")
            output_f.close()


            
if __name__ == "__main__":
    config = yaml.safe_load(open("./config.yaml"))
    paras = get_cmd()
    for k, v in paras.__dict__.items():
        config[k] = v
    config = config_dataset(config)
    print(config)
    train_eval(config)
