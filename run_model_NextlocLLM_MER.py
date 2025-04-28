import argparse

from libcity.pipeline import run_model_NextlocLLM_MER_lora,test_model_NextlocLLM_MER_lora
from libcity.utils import str2bool, add_general_args

def is_accelerate_runtime():
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    #配置文件
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    #随机种子
    parser.add_argument('--mer_dim',  type=int, default=128)
    #坐标embedding的维度
    parser.add_argument('--day_dim',  type=int, default=16)
    #时间embedding的维度
    parser.add_argument('--hour_dim',  type=int, default=16)
    #小时embedding的维度
    parser.add_argument('--dur_dim',  type=int, default=16)
    #持续时间embedding的维度
    parser.add_argument('--if_train',  type=int, default=1)
    #训练还是测试
    parser.add_argument('--save_dir',type=str)
    #参数保存路径
    parser.add_argument('--save_dict',type=str)
    #参数保存路径
    parser.add_argument('--his_seq_len',type=int,default=40)
    parser.add_argument('--cur_seq_len',type=int,default=5)
    parser.add_argument('--num_experts_embedding',type=int,default=5)
    parser.add_argument('--expert_freq',type=int,default=2)
    #输入部分有多少embedding
    parser.add_argument('--max_len_poi',type=int,default=60)
    parser.add_argument('--num_frozen_layer',type=int,default=8)
    parser.add_argument('--total_usage_layers',type=int,default=12)
    parser.add_argument('--num_persona_experts',type=int,default=11)
    parser.add_argument('--routing',type=str,default='threshold')
    parser.add_argument('--threshold',type=float,default=0.8)
    parser.add_argument('--lambda_',type=float,default=300.0)
    # 增加其他可选的参数
    add_general_args(parser)
    # 解析参数
    args = parser.parse_args()

    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    #print(args.if_prompt)
    if(args.if_train):
        run_model_NextlocLLM_MER_lora(
                        config_file=args.config_file, 
                        other_args=other_args)
    else:
        test_model_NextlocLLM_MER_lora(
                        config_file=args.config_file, 
                        other_args=other_args)
