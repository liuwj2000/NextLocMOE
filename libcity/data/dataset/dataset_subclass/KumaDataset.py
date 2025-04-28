
import os
import json
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
import importlib
from logging import getLogger
import pickle
import gc

from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_timeoff
from libcity.data.utils import generate_dataloader_pad




def load_history_data(name,type):
    file_dir='./data/llmmob/Kumamoto/history_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data

def load_content_data(name,type):
    file_dir='./data/llmmob/Kumamoto/context_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data

def load_content_data_true(name,type):
    file_dir='./data/llmmob/Kumamoto/context_true_'+name+'_'+type+'.pkl'
    #print(file_dir)
    f=open(file_dir,'rb')
    data=pickle.load(f)
    #print('load file: ',f,' finished')
    f.close()
    return data


class KumaDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.train_data_dir=''
        self.logger = getLogger()

    def get_data(self):
        

        #####################################################            load history data and related info       #####################################################

        history_data_train=load_history_data('data','train')
        history_dur_train=load_history_data('dur','train')
        history_hour_train=load_history_data('hour','train')
        history_day_train=load_history_data('day','train')       

        history_data_valid=load_history_data('data','vali')
        history_dur_valid=load_history_data('dur','vali')
        history_hour_valid=load_history_data('hour','vali')
        history_day_valid=load_history_data('day','vali')

        history_data_test=load_history_data('data','test') 
        history_dur_test=load_history_data('dur','test')
        history_hour_test=load_history_data('hour','test')
        history_day_test=load_history_data('day','test')


        ###############################################################################################################################################################
        
        


        #####################################################            load content data and related info   for training     #####################################################
        context_data_train=load_content_data('data','train')
        context_dur_train=load_content_data('dur','train')
        context_hour_train=load_content_data('hour','train')
        context_day_train=load_content_data('day','train')
        
        ###############################################################################################################################################################
        




        #####################################################            load content data and related info   for training     #####################################################
        context_data_valid=load_content_data('data','vali')
        context_dur_valid=load_content_data('dur','vali')
        context_hour_valid=load_content_data('hour','vali')
        context_day_valid=load_content_data('day','vali')
        ###############################################################################################################################################################
        



        #####################################################            load content data and related info   for training     #####################################################
        context_data_test=load_content_data('data','test')
        context_dur_test=load_content_data('dur','test')
        context_hour_test=load_content_data('hour','test')
        context_day_test=load_content_data('day','test')
        ###############################################################################################################################################################





        ####################################################            load ground truth data  ############################################################

        context_data_train_true=load_content_data_true('data','train')
        context_dur_train_true=load_content_data_true('dur','train')
        context_data_idx_train_true=load_content_data_true('data_idx','train')

        context_data_valid_true=load_content_data_true('data','vali')
        context_dur_valid_true=load_content_data_true('dur','vali')
        context_data_idx_valid_true=load_content_data_true('data_idx','vali')

        context_data_test_true=load_content_data_true('data','test')
        context_dur_test_true=load_content_data_true('dur','test')
        context_data_idx_test_true=load_content_data_true('data_idx','test')
        #####################################################################################################################################################
        
        
        
        
        
        ###########################################   train_data ebegin ###########################################
        train_data_llm=[]
        for user in history_data_train:
            len_user=len(context_data_train[user])
            for i in range(0,len_user,20):
                tmp=[]
                tmp.append(history_data_train[user][i])
                tmp.append(history_day_train[user][i]) 
                tmp.append(history_hour_train[user][i])              
                tmp.append(history_dur_train[user][i])
                
                tmp.append(context_data_train[user][i])
                tmp.append(context_day_train[user][i])
                tmp.append(context_hour_train[user][i])
                tmp.append(context_dur_train[user][i])
                
                tmp.append(context_data_train_true[user][i])
                tmp.append(context_dur_train_true[user][i])
                tmp.append(context_data_idx_train_true[user][i])
                tmp.append(user)
                train_data_llm.append(tmp)
        self.logger.info('Num of training data:{}'.format(len(train_data_llm)))
        ###########################################   train_data end ###########################################




        ###########################################   valid_data begin ###########################################
        valid_data_llm=[]
        for user in history_data_valid:
            len_user=len(context_data_valid[user])
            for i in range(0,len_user,20):
                tmp=[]
                tmp.append(history_data_valid[user][i])
                tmp.append(history_day_valid[user][i])
                tmp.append(history_hour_valid[user][i])
                tmp.append(history_dur_valid[user][i])
                
                tmp.append(context_data_valid[user][i])
                tmp.append(context_day_valid[user][i])
                tmp.append(context_hour_valid[user][i])
                tmp.append(context_dur_valid[user][i])
                
                tmp.append(context_data_valid_true[user][i])
                tmp.append(context_dur_valid_true[user][i])
                tmp.append(context_data_idx_valid_true[user][i])
                tmp.append(user)
                valid_data_llm.append(tmp)

        self.logger.info('Num of validation data:{}'.format(len(valid_data_llm)))
        ###########################################   valid_data end ###########################################




        ###########################################   test_data begin ###########################################
        test_data_llm=[]
        for user in history_data_test:
            len_user=len(context_data_test[user])
            for i in range(0,len_user,20):
                tmp=[]
                tmp.append(history_data_test[user][i])
                tmp.append(history_day_test[user][i])
                tmp.append(history_hour_test[user][i])
                tmp.append(history_dur_test[user][i])
                
                tmp.append(context_data_test[user][i])

                tmp.append(context_day_test[user][i])
                tmp.append(context_hour_test[user][i])
                tmp.append(context_dur_test[user][i])
                #tmp.append(context_poi_test[user][i])
                
                tmp.append(context_data_test_true[user][i])
                tmp.append(context_dur_test_true[user][i])
                tmp.append(context_data_idx_test_true[user][i])
                tmp.append(user)
                test_data_llm.append(tmp)

        self.logger.info('Num of testing data:{}'.format(len(test_data_llm)))
        ###########################################   test_data end ###########################################
        
        
        self.loc_x_mean=9807.908794318473
        self.loc_y_mean=8113.654011362933

        self.loc_x_std=22859.94461536162
        self.loc_y_std=20353.834963293164
        self.duration_max=24
        self.duration_min=0.1
        
        feature_dict={'history_loc':'array of int',
                      'history_day':'int',
                      'history_hour':'int',
                      'history_dur':'int',
                      'current_loc':'array of int',
                      'current_day':'int',
                      'current_hour':'int',
                      'current_dur':'int',
                      'target'     :'array of int',
                      'target_dur' :'int',
                      'target_idx':'int',
                      'uid'        :'int'}

        pad_item={}

        # num of stations 


        train_dataloader,valid_data_loader,test_dataloader=generate_dataloader_pad(train_data_llm, 
                                       valid_data_llm,     
                                       test_data_llm,
                                       feature_dict,
                                       self.config['batch_size'],
                                       self.config['num_workers'], 
                                       pad_item
                                       )
        print(f"#Train batches: {len(train_dataloader)}")
        print(f"#Valid batches: {len(valid_data_loader)}")
        print(f"#Test batches: {len(test_dataloader)}")

            
        return train_dataloader,valid_data_loader,test_dataloader

    def get_data_feature(self):
        res = {'loc_size':  332,
               'day_size':  7,
               'hour_size': 24,
               'loc_x_mean':self.loc_x_mean,
               'loc_y_mean':self.loc_y_mean,
               'loc_x_std':self.loc_x_std,
               'loc_y_std':self.loc_y_std,
               'duration_max':self.duration_max,
               'duration_min':self.duration_min}
        #res['distance_upper'] = self.config['distance_upper']
        return res

    
