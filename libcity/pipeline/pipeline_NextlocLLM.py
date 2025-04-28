import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
import pandas as pd
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
from accelerate import Accelerator
from torch import nn, optim
import numpy as np
import shutil
from tqdm import tqdm
import time
import datetime
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate.utils import set_seed
from scipy.spatial import KDTree
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import matplotlib
import gc
matplotlib.use('Agg')


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 自动 strip 掉 'module.' 前缀
def strip_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def top_k(loc_pred, loc_true, topk,tree,accelerator):
    loc_pred = loc_pred.numpy()
    hit = 0
    result=tree.query(loc_pred,k=topk)[1]
    #batch,topk
    #accelerator.print('result',result,'*',loc_pred)
    for i in range(len(loc_true)):
        #accelerator.print('result in',result[i],loc_true[i])
        if isinstance(result[i], np.int64):
            if(loc_true[i]==result[i]):
                hit+=1
        else:
            if(loc_true[i] in  result[i]):
                hit+=1
    return hit

def vali_test(config, accelerator,  model, vali_loader,tree,flag):
    model.eval()

    total_samples = 0
    total_distance = 0.0
    hit_k = {1: 0, 5: 0, 10: 0, 20: 0}

    with torch.no_grad():
        for i, batch in enumerate(vali_loader):
            _, pred_coords, _,_ = model(batch, accelerator)
            true_coords = torch.as_tensor(batch['target']).float().to(accelerator.device)

            # 均方距离
            total_distance += torch.linalg.norm(pred_coords - true_coords, dim=1).sum().item()
            total_samples += len(batch['target_idx'])

            
            pred_coords = pred_coords.cpu().detach()
            target_indices = batch['target_idx']

            for k in hit_k.keys():
                hit_k[k] += top_k(pred_coords, target_indices, k, tree, accelerator)

    model.train()
    # === 打印评估结果 ===
    if accelerator.is_main_process:
        accelerator.print(f"[{flag.upper()}] Evaluation Results:")
        for k in hit_k:
            accelerator.print(f"  ➤ Hit@{k}: {hit_k[k] / total_samples:.6f} (hit: {hit_k[k]}, total: {total_samples})")
        accelerator.print(f"  ➤ Avg. Distance Error: {total_distance / total_samples:.6f} meters")
        accelerator.print("-" * 60)

    return( hit_k[1] / total_samples,
        hit_k[5] / total_samples,
        hit_k[10] / total_samples,
        total_distance / total_samples)

def initialize_config(config_file, other_args):
    #配置加载&设置
    config = ConfigParser('traj_loc_pred', 
                        'NextlocLLM_MER_lora', 
                        'METR_LA',
                        config_file=config_file,
                        other_args=other_args)

    seed = config.get('seed', 0)
    #随机种子设置
    set_seed(seed)

    return config

def create_KD_tree(config):
    topk_file = config["topk_file"]
    loc_ = pd.read_csv(topk_file)
    coordinates = loc_[['mercator_x_subzone', 'mercator_y_subzone']].values

    #print('coord',coordinates)
    '''
    [[12947162.07049679  4863015.17602583]
    [12949214.279517    4866146.65995776]
    [12949713.05344352  4864807.14102297]
    ...
    [12950718.66943083  4857258.7764302 ]
    [12947680.11898755  4866494.98623617]
     [12951917.82386099  4846345.21855565]]
    '''

    tree = KDTree(coordinates)
    # validation 相关的 KD树

    return tree

def create_dataset(config):

    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()




    return train_data, valid_data, test_data,data_feature

def prepare_model(config,train_data, valid_data, test_data,data_feature, accelerator):

    model = get_model(config,data_feature)
    #创建模型

    
    #创建accelerator
    accelerator.print(model)
    accelerator.print(config.config)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
    #设置可训练参数

    for name, param in model.named_parameters():
        if param.requires_grad:
            accelerator.print(f"Parameter: {name}")       
    #需要train的参数【部分参数被冻结了，不用train】

    model_optim = optim.Adam(trained_parameters, 
                            lr=config['learning_rate'],
                            weight_decay=0.01)
    #学习率更新


    train_loader, vali_loader, test_loader, model, model_optim= accelerator.prepare(
            train_data, 
            valid_data, 
            test_data, 
            model, 
            model_optim)

    return train_loader, vali_loader, test_loader, model, model_optim,accelerator

def log_grad_norms(model, accelerator, norm_type=2):
    total_norm = 0.0
    flag=0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            if(param_norm.item() >1000000):
                flag=1
                accelerator.print(f"Grad norm for {name}: {param_norm.item():.6f}")
    if(flag==1):
        total_norm = total_norm ** (1. / norm_type)
        accelerator.print(f"Total gradient norm: {total_norm:.6f}")

def train_and_valid(config, accelerator, model, model_optim, train_loader, vali_loader, test_loader, tree):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    #防止内存碎片化导致 OOM
    best_val=-1
    #最好的验证loss

    scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=0.5, 
        patience=2, verbose=True)
    
    lst_train_loss=[]
    lst_dis_loss=[]
    lst_dur_loss=[]

    lst_loss_tst=[]
    hit_rate_tst=[]
    lst_loss_val=[]
    lst_loss_tr=[]

    for epoch in range(config['max_epoch']):
        model.train()
        train_loss, dis_loss, dur_loss = [], [], []
        epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                model_optim.zero_grad()

                # 在使用 accelerate 库的情况下正确调用自定义方法
                loss,distance_loss,du_loss = model.module.calculate_loss(batch,accelerator)
                
                if torch.isnan(loss):
                    accelerator.print(batch)
                    accelerator.print("❌ Loss is NaN. Skipping this batch.")
                    continue

                accelerator.backward(loss)
                log_grad_norms(model,accelerator)
                # 打印当前所有可训练参数的梯度范数（L2 norm）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2500.0)
                model_optim.step()

                train_loss.append(loss.item())
                dis_loss.append(distance_loss.item())
                dur_loss.append(du_loss.item())
                
        
        scheduler.step(np.mean(train_loss))

        


        hit_rate1,hit_rate5,hit_rate10,tst_loss= vali_test(config, accelerator, model, test_loader,tree,'test')
        hit_rate1_v,hit_rate5_v,hit_rate10_v,tst_loss_v = vali_test(config, accelerator, model, vali_loader,tree,'valid')
        #hit_rate1_tr,hit_rate5_tr,hit_rate10_tr,tst_loss_tr = vali_test(config, accelerator, model, train_loader,tree,'train')
        #当前epoch的train valid 和test的hit rate

        current_lr = model_optim.param_groups[0]['lr']
        accelerator.print("Epoch: {} cost time: {} lr {}".format(epoch + 1, time.time() - epoch_time,current_lr))
        accelerator.print(f"Epoch {epoch + 1} | Train Loss: {np.mean(train_loss):.7f} "
                          f"test_hit@10: {hit_rate10:.7f} distance: {np.mean(dis_loss):.7f} duration: {np.mean(dur_loss):.7f}")


        if accelerator.is_main_process and hit_rate10_v > best_val:
            best_val = hit_rate10_v
            unwarp_model = accelerator.unwrap_model(model)
            torch.save(unwarp_model.state_dict(), config['save_dict'])

        # epoch 末清理
        torch.cuda.empty_cache()
        gc.collect()

    hit_rate1,hit_rate5,hit_rate10,tst_loss = vali_test(config, accelerator, model, test_loader,tree,'test')
    print("Finally test_hit_rate_1: {0:.7f} test_hit_rate_5:{1:.7f} test_hit_rate_10:{2:.7f} ".format(
                    hit_rate1,hit_rate5,hit_rate10,model.device))

def run_model_NextlocLLM_MER_lora(config_file=None, other_args=None):

    
    config=initialize_config(config_file,other_args)
    #加载配置
    
    tree=create_KD_tree(config)
    #创建KD树

    accelerator = Accelerator(gradient_accumulation_steps=4)
    #创建accelerator
    
    train_data, valid_data, test_data,data_feature=create_dataset(config)
    #创建数据集

    train_loader, vali_loader, test_loader, model, model_optim,accelerator=prepare_model(config,train_data, valid_data, test_data,data_feature, accelerator)
    '''
    model_path = config.get('resume_path', '/home_nfs/liushuai/NextlocMOE/acce_file/NextlocMOE.pth')
    if os.path.exists(model_path):
        accelerator.print(f"✅ Resuming from checkpoint: {model_path}")
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = torch.load(model_path, map_location=accelerator.device)
        state_dict = strip_module_prefix(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=False)
        accelerator.print("✅ Model weights loaded. Continuing training...")
    '''


    train_and_valid(config, accelerator, model, model_optim, train_loader, vali_loader, test_loader, tree)
 

def test_model_NextlocLLM_MER_lora(config_file=None,
                        other_args=None):
    
    initialize_config(config_file, other_args)
    #加载配置

    tree=create_KD_tree(config)
    #创建KD树

    train_data, valid_data, test_data,data_feature=create_dataset(config)
    #创建数据集


    model = get_model(config,data_feature)

    accelerator = Accelerator()

    train_loader, vali_loader, test_loader, model= accelerator.prepare(
            train_data, 
            valid_data, 
            test_data, 
            model)
    
    accelerator.load_state(config["save_dir"])
    train_steps = len(train_loader)

    loc_loss_lst=[]
    dur_loss_lst=[]
    model.eval()


    with torch.no_grad():
            hit_rate1,hit_rate5,hit_rate10,tst_loss = vali_test(config, accelerator, model, test_loader,tree)
            print(
                "Finally test_hit_rate_1: {0:.7f} test_hit_rate_5:{1:.7f} test_hit_rate_10:{2:.7f} ".format(
                    hit_rate1,hit_rate5,hit_rate10))
            
    
        