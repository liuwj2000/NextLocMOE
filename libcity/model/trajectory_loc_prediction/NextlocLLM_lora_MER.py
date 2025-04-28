import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from logging import getLogger
from torch.utils.checkpoint import checkpoint
from libcity.model.abstract_model import AbstractModel
import pandas as pd
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from pytorch_tcn import TCN
from transformers import AutoModel,AutoTokenizer
from torch.nn.utils import remove_weight_norm
import copy
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

#torch.set_printoptions(threshold=5000)

os.environ["HF_TOKEN"] = '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'



class MoEBlock(nn.Module):
    def __init__(self, input_dim, history_dim,num_experts, original_ffn,routing_strategy,threshold):
        super().__init__()

        # 预设 LoRA 配置
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=16,
            target_modules=["gate_proj", "up_proj", "down_proj"],  # 根据LLaMA3 FFN模块定义来
            lora_dropout=0.02,
            bias="none"
        )

        self.experts = nn.ModuleList([
            get_peft_model(copy.deepcopy(original_ffn), lora_config)
            for _ in range(num_experts)])          
        '''
            #获取原始前馈网络（original_ffn）的类类型

            #使用原始网络的配置参数初始化新实例
        ''' 
        #nn.ModuleList：将生成的专家打包为PyTorch可识别的模块列表
        
        fusion_input_dim = input_dim + history_dim + input_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim)
        )

        self.gate = nn.Linear(input_dim, 1)
        self.routing_strategy=routing_strategy
        self.num_experts=num_experts
        
        self.threshold=threshold

    def _expert_selection(self, gate_probs,k=2):
        #gate_prob batch,seq_len,num_expert
        if self.routing_strategy == 'topk':
            # 取top2专家
            topk_values, topk_indices = torch.topk(gate_probs, k, dim=-1)
            #batch,seq_len,2
            mask = torch.zeros_like(gate_probs).scatter_(-1, topk_indices, 1).bool()
            '''
            构建二进制掩码（选中位置为True）
            初始全零矩阵 batch,seq_len,num_expert， topk_indices对应位置填入1（True）
            '''
            sorted_indices = torch.where(mask, torch.arange(gate_probs.size(-1), device=gate_probs.device), -1)
            sorted_probs = torch.where(mask, gate_probs, torch.zeros_like(gate_probs))


        elif self.routing_strategy == 'threshold':
            sorted_probs, sorted_indices = torch.sort(gate_probs, descending=True, dim=-1)
            # 步骤1: 对专家概率降序排列
            #sorted_probs batch,seq_len,num_expert
            
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 步骤2: 计算累积概率
            #sorted_probs batch,seq_len,num_expert
            
            mask = cumulative_probs > self.threshold
            #batch,seq_len,num_expert
            # 步骤3: 找出首个超过阈值的索引

            threshold_index=mask.long().argmax(dim=-1)
            #batch,seq_len

            threshold_mask=torch.nn.functional.one_hot(threshold_index,
                num_classes=sorted_probs.size(-1)).bool()
            #batch,seq_len,num_expert
            #只有首个超过阈值（也就是mask是true的位置）是True，其他位置都是false
            
            mask=mask & ~ threshold_mask
            #batch,seq_len,num_expert
            '''
            掐断top-p之后的位置 [0,0,0,0,1,1,1,1]
            '''

            #activated_count = mask.sum(dim=-1)  # => [B, S]
            #print(f"🔢 before Activated expert counts (per token):\n{activated_count}")

            #最多一次性选5个专家
            
            top_k_limit = 5
            cumsum_mask = torch.cumsum(mask.float(), dim=-1)
            mask = mask & (cumsum_mask <= top_k_limit)
            
            expand_mask = threshold_mask.gather(-1, sorted_indices.argsort(-1))
            # 步骤5: 将掩码映射回原始排序
            # sorted_indices.argsort(-1) 获得原始位置的索引映射
            #通过gather将阈值掩码还原到原始顺序
            
            mask = expand_mask & (gate_probs > 0)
            #print(mask)
            # 步骤6: 过滤零概率专家

            sorted_indices=torch.where(mask,-1,sorted_indices)
            sorted_probs=torch.where(mask,0.0,sorted_probs)
            
            #activated_count = mask.sum(dim=-1)  # => [B, S]
            #print(f"🔢 after Activated expert counts (per token):\n{activated_count}")


        return sorted_probs, sorted_indices
           
    def forward(self, x,history_hidden_embedding,persona_embedding, return_entropy=False):
        '''
        x:batch,seq_len,llm_dim
        history_hidden_embedding：batch_size,self.total_input_dim
        persona_embedding:10,50,llm_dim
        '''
        # 专家门控
        B, S, D = x.shape
        E, D = persona_embedding.shape  # persona experts


        # 2. 准备 gate logits：我们要构建一个 (batch, seq_len, num_expert) 的张量
        gate_logits = []
        for i in range(self.num_experts):
            expert_persona = persona_embedding[i].unsqueeze(0).unsqueeze(0)  # (1, 1, llm_dim)
            expert_persona_expand = expert_persona.expand(B, S, -1)     # (B, S, D)

            history_expand = history_hidden_embedding.unsqueeze(1).expand(-1, S, -1)  
            # (B, S, total_input_dim)

            gate_input = torch.cat([x, history_expand, expert_persona_expand], dim=-1)  
            # (B, S, total_input_dim+2*llm_dim)

            fused = self.fusion_layer(gate_input)  
            # (B, S, hidden_dim)
            logit = self.gate(fused).squeeze(-1)   
            # (B, S)，表示对第 i 个专家的评分
        
            gate_logits.append(logit)

        gate_logits = torch.stack(gate_logits, dim=-1)
        # 拼接所有专家评分 → [B, S, E]

        
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        topk_weights, topk_ind = self._expert_selection(gate_probs)
        #得到掩码专家
        
        
        #  将输入 x、routed_probs 以及 mask 扁平化，方便后续对各 token 进行处理
        B, S, D = x.shape
        x_flat = x.view(-1, D) 
        # [B*S, D] 
        topk_weights = topk_weights.view(-1, topk_weights.size(-1))  
        # [B*S, E]
        topk_ind = topk_ind.view(-1, topk_ind.size(-1))  
        # [B*S, E]
        
        
        
        # 初始化输出缓冲区
        output_total = torch.zeros_like(x_flat,
                                      device=x.device,
                                      dtype=x.dtype)  
        # [B*S, D]
        
        # 5. 遍历每个专家，根据 mask 选择需要处理的 token，并计算输出加权累加
        for expert_num, expert in enumerate(self.experts):
            
            sample_ind,expert_ind=torch.where(topk_ind==expert_num)
            #sample_ind [0~B*S]; expert [0~E]
            hidden=x_flat[sample_ind,:]
            #从[s*b,h]的token表示中拿出当前专家要处理的token，(n,h)维度

            expert_out=expert(hidden)
            #(n,h)

            output_total[sample_ind]+=torch.mul(expert_out,
                topk_weights[sample_ind,expert_ind].unsqueeze(1))
            #将专家输出加权写到总输出中

        output_total=output_total.view(B,S,D)
        if return_entropy:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-9)).sum(dim=-1).mean()  # scalar
            return output_total, entropy
        else:
            return output_total
     
class EnhancedLLM(nn.Module):  
    def __init__(self,config,history_dim):
        super().__init__()
        self._init_config(config,history_dim)
        self._init_model()
        self._apply_freezing()
        self.convert_to_moe()
        
    def _init_config(self,config,history_dim):
        self.total_usage_layers=config['total_usage_layers']
        self.num_frozen_layer=config['num_frozen_layer']
        self.expert_freq=config['expert_freq']
        self.history_dim=history_dim
        self.num_persona_experts=config['num_persona_experts']
        self.routing=config['routing']
        self.threshold=config['threshold']
        
        
    def _init_model(self):
        self.tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
        self.model=AutoModel.from_pretrained('meta-llama/Llama-3.2-3B',
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True,
                                                num_hidden_layers=self.total_usage_layers) 
        self.llm_dim=self.model.get_input_embeddings().weight.shape[1]
        if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

        #self.model.gradient_checkpointing_enable()
        '''
        只要模块是标准 nn.Module，就能被 torch.utils.checkpoint 显式包裹

        expert 是 get_peft_model(...) 得到的 LoRA 模型，也是 nn.Module

         MoEBlock 是在 LLaMA transformer 的 mlp 位置被替换的，而又启用了 .gradient_checkpointing_enable()
        ——> 整个 transformer layer 会以 block 为单位 checkpoint
        ——>MoEBlock（包含 experts 和 fusion_layer）会被一整个 forward graph checkpoint 掉（不用手动包checkpoint）
        '''
        
    def _apply_freezing(self):
            for param in self.model.embed_tokens.parameters():
                param.requires_grad_(False)
            for layer in self.model.layers:
                for param in layer.self_attn.parameters():
                    param.requires_grad_(False)
                    # 冻结所有注意力层（全模型范围）
                for param in layer.mlp.parameters():
                    param.requires_grad_(False)
                    #冻结所有 FFN 层（先全冻）,再在convert_to_moe中，把要转化成MOE的mlp层解冻          

    def convert_to_moe(self):
        start = self.num_frozen_layer
        end = self.total_usage_layers
        freq=self.expert_freq
        for layer_idx in range(start,end,freq):
            original_layer = self.model.layers[layer_idx] 
            original_ffn = original_layer.mlp           
            moe_block=MoEBlock(self.llm_dim, 
                self.history_dim,
                self.num_persona_experts,
                original_ffn,
                self.routing,
                self.threshold)
            original_layer.mlp = moe_block         

    def forward(self,hidden_states,
        position_ids=None,
        history_hidden_embedding=None,
        persona_embedding=None,):
        '''
        x: (B, S,D)
        history_hidden_embedding: (B, D_h)
        persona_embedding: (E, T, D)
        '''

        bsz, seqlen,_ = hidden_states.shape
        device =hidden_states.device

        # === 2. Rotary Embedding ===
        if position_ids is None:
            position_ids = torch.arange(seqlen, 
                    dtype=torch.long, 
                    device=device).unsqueeze(0).expand(bsz, -1)
        

        entropy_losses = []
        
        
        # === 3. Causal Mask ===
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            
            mask = mask.to(dtype=hidden_states.dtype)

        # === 4. Transformer Layers ===
        for layer in self.model.layers:
            # --- Attention ---
            residual = hidden_states

            #freqs_cis = layer.self_attn.rotary_emb(position_ids).to(device)

            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(
                hidden_states,
                position_ids=position_ids,
                mask=mask
            )
            # 返回attn_output, attn_weights
            hidden_states = residual + hidden_states[0]  
            # Add & Norm

            # --- FFN (MoE or standard) ---
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            if isinstance(layer.mlp, MoEBlock):
                # MoE Block：接收额外信息
                hidden_states, entropy = layer.mlp(
                    hidden_states,
                    history_hidden_embedding,
                    persona_embedding,
                    return_entropy=True
                )
                entropy_losses.append(entropy)
            else:
                # 普通 FFN
                hidden_states = layer.mlp(hidden_states)

            hidden_states = residual + hidden_states  # Add & Norm
        return hidden_states,entropy_losses

class POIEmb_prob(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(POIEmb_prob,self).__init__()

        self.fc=nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,out_dim)
        )

    def forward(self,x):
        return self.fc(x)

class Normalize_loc(nn.Module):
    def __init__(self,loc_x_mean,loc_y_mean,loc_x_std,loc_y_std):
        super(Normalize_loc,self).__init__()
        self.loc_x_mean=loc_x_mean
        self.loc_y_mean=loc_y_mean
        self.loc_x_std=loc_x_std
        self.loc_y_std=loc_y_std
        self.loc_mean=torch.FloatTensor([self.loc_x_mean,self.loc_y_mean])
        self.loc_std=torch.FloatTensor([self.loc_x_std,self.loc_y_std])
        #print(self.loc_mean,self.loc_std)
    def forward(self,x,mode):
        if mode=='norm':
            x=self.normalize(x)
        elif mode=='denorm':
            x=self.denormalize(x)
        return x

    def normalize(self,x):
        self.loc_mean=self.loc_mean.to(x.device)
        self.loc_std=self.loc_std.to(x.device)
        x=(x-self.loc_mean)/self.loc_std
        return x
    
    def denormalize(self,x):
        self.loc_mean=self.loc_mean.to(x.device)
        self.loc_std=self.loc_std.to(x.device)
        x=x*self.loc_std+self.loc_mean
        return x 

class Normalize_dur(nn.Module):
    def __init__(self,duration_max,duration_min):
        super(Normalize_dur,self).__init__()
        self.duration_max=torch.FloatTensor([duration_max])
        self.duration_min=torch.FloatTensor([duration_min])
        #print(self.duration_max,self.duration_min)
    def forward(self,x,mode):
        if mode=='norm':
            x=self.normalize(x)
        elif mode=='denorm':
            x=self.denormalize(x)
        return x

    def normalize(self,x):
        self.duration_min=self.duration_min.to(x.device)
        self.duration_max=self.duration_max.to(x.device)
        x=(x-self.duration_min)/(self.duration_max-self.duration_min)
        return x
    
    def denormalize(self,x):
        self.duration_min=self.duration_min.to(x.device)
        self.duration_max=self.duration_max.to(x.device)
        x=x*(self.duration_max-self.duration_min)+self.duration_min
        return x

class NextlocLLM_MER_lora(AbstractModel):
    """rnn model with long-term history attention"""

    def __init__(self, config, data_feature):
        super(NextlocLLM_MER_lora, self).__init__(config, data_feature) 
        self.config=config
        self.data_feature=data_feature
        #####################  new for NextLocMOE####################################
        

        self.__init_config__()
        self.__init_embeddings__()
        self.__init_normalizer()
        self.__init_TCN__()
        self.__init_LLM__()
        self.__init_POI_embedding__()
        self.__init_current_POIEmb_prob()
        self.__init_input_NextMOE_()
        self.__init_output_NextMOE_()
        self.__init_prompt()
        self.__init_persona_prompt()

    def __init_config__(self):
        #self.dropout = nn.Dropout(p=self.config['dropout_p'])
        self.his_seq_len=self.config['his_seq_len']
        self.cur_seq_len=self.config['cur_seq_len']

        self.num_experts_embedding=self.config['num_experts_embedding']
        self.max_len_poi=60

    def __init_embeddings__(self):
        #data location embedding
        self.mer_size = 2
        self.mer_dim=self.config['mer_dim']
        self.mer2vec=nn.Linear(self.mer_size,self.mer_dim,bias=False)
        #(batch,seq_len,2)——>(batch,seq_len,mervec_size)
        
        #day embedding 
        self.day_size = self.data_feature['day_size']
        self.day_dim=self.config['day_dim']
        self.day_embedding=nn.Embedding(self.day_size,
                                        self.day_dim)
        #(batch,seq_len)——>(batch,seq_len,7)
  
        #hour embedding 
        self.hour_size=self.data_feature['hour_size']
        self.hour_dim=self.config['hour_dim']
        self.hour_embedding=nn.Embedding(self.hour_size,
                                        self.hour_dim)
        #(batch,seq_len)——>(batch,seq_len,hour_emb_size)
        
        #duration Linear
        self.dur_dim=self.config['dur_dim']
        self.dur_linear=nn.Linear(1,self.dur_dim,bias=False)

        self.total_input_dim=self.mer_dim+self.day_dim+self.hour_dim+self.dur_dim       
    def __init_normalizer(self):
        self.normalize_loc=Normalize_loc(self.data_feature['loc_x_mean'],
                                        self.data_feature['loc_y_mean'],
                                        self.data_feature['loc_x_std'],
                                        self.data_feature['loc_y_std'])

        self.normalize_dur=Normalize_dur(self.data_feature['duration_max'],
                                        self.data_feature['duration_min'])
    def __init_TCN__(self):
        self.tcn=TCN(
            num_inputs=self.total_input_dim,
            num_channels=[self.total_input_dim,
                self.total_input_dim,
                self.total_input_dim,
                self.total_input_dim,
                self.total_input_dim],
            input_shape='NLC')
        self.tcn_ln = nn.LayerNorm(self.total_input_dim)
    def __init_LLM__(self):
        
        self.model=EnhancedLLM(self.config,self.total_input_dim)
        self.tokenizer=self.model.tokenizer
        '''
        self.tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
        self.model=AutoModel.from_pretrained('meta-llama/Llama-3.2-3B',
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True,
                                                num_hidden_layers=self.config['total_usage_layers']
                                                ) 
        '''
        self.llm_dim=self.model.llm_dim
        #self.llm_dim=self.model.get_input_embeddings().weight.shape[1]

        '''
        if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

        for param in self.model.embed_tokens.parameters():
                param.requires_grad_(False)
        for layer in self.model.layers:
                for param in layer.self_attn.parameters():
                    param.requires_grad_(False)
                    # 冻结所有注意力层（全模型范围）
                for param in layer.mlp.parameters():
                    param.requires_grad_(False)
                    #冻结所有 FFN 层（先全冻）,再在convert_to_moe中，把要转化成MOE的mlp层解冻      
        '''  
    def __init_POI_embedding__(self):    
        category_descriptions = {
            'Entertainment':'This category includes scenic spots, sports venues, and recreational facilities, offering activities for leisure, entertainment, and social interactions.Typical examples include amusement parks, cinemas, stadiums, and bars. Users often visit for relaxation, nightlife, sports, and cultural experiences, with peak times in evenings and weekends.',
            'Commercial':'This category encompasses businesses, financial institutions, automotive services, shopping centers, and dining establishments, supporting daily consumer and professional needs. Typical examples include malls, banks, car dealerships, and restaurants. Users often visit during working hours or weekends for shopping, financial transactions, or dining.',
            'Education':'This category covers institutions focused on academic, cultural, and scientific learning. Typical examples include schools, universities, libraries, and research centers. Users often visit on weekdays for study, teaching, research, and cultural enrichment.',
            'Public':'This category includes government offices, healthcare facilities, transportation hubs, and other essential public infrastructure. Typical examples include city halls, hospitals, bus stations, and utility centers. Users often visit for administrative tasks, medical needs, commuting, or essential services, with varied peak hours depending on the service type.',
            'Residential':'This category comprises housing areas, mixed-use developments, and temporary accommodations. Typical examples include apartment complexes, residential neighborhoods, and hotels. Users often visit for long stays, typically peaking in the evenings, weekends, and holidays.'
        }

        with torch.no_grad():
            text_embeddings = []
            for desc in category_descriptions.values():
                encoded=self.tokenizer(desc,
                            return_tensors="pt",
                            padding='max_length', 
                            max_length=self.max_len_poi,
                            truncation=True).input_ids
                emb=self.model.model.embed_tokens(encoded)
                #emb=self.model.embed_tokens(encoded)
                text_embeddings.append(emb.mean(dim=[0,1]))
                #(1,len_prompt_seq,llm_emb)->(llm_emb)

        #self.poi_to_emb_mlp=nn.Linear(self.llm_dim,self.mer_dim)

        self.poi_to_emb_mlp = nn.Sequential(
            nn.Linear(self.llm_dim, self.mer_dim),
            nn.LayerNorm(self.mer_dim)
        )

        self.category_mer2vec = nn.ModuleList()
        
        for emb in text_embeddings:
            # 投影到mer_dim空间
            vec = self.poi_to_emb_mlp(emb.float())
            #(mer_dim)
            
            mlp = nn.Linear(self.mer_size, self.mer_dim, bias=False)
            # 创建地理编码层

            # 用vec复制两次初始化权重矩阵
            with torch.no_grad():
                # 权重形状应为 (input_dim, output_dim) = (2, mer_dim)
                mlp.weight.data = vec.repeat(2, 1).T  
                # (mer_dim, 2) 的转置
                
            self.category_mer2vec.append(mlp)
            #(num_poi,mer_size,mer_dim)           
    def __init_current_POIEmb_prob(self):
        output_dim_prob=5
        input_dim_prob=2*self.total_input_dim

        self.poi_emb_prob=POIEmb_prob(input_dim_prob,output_dim_prob)                                           
    def __init_input_NextMOE_(self):                
        self.history_llmize_linear=nn.Linear(self.total_input_dim,
                                            self.llm_dim,
                                            bias=False)
        self.current_llmize_linear=nn.Linear(self.total_input_dim,
                                            self.llm_dim,
                                            bias=False)     
    def __init_output_NextMOE_(self):

        self.output_dur=nn.Linear(self.llm_dim,1,bias=False)
        self.output_loc=nn.Linear(self.llm_dim,2,bias=False)
    def __init_prompt(self):
        prompt=(
                    f"<|start_prompt|>Task Description: Predict the next possible location, in normalized mercator coordinates, of a resident based on their historical and current movement trajectory. "
                    f"Data Description: This dataset includes mobility trajectory data of residents. "
                    f"Each record consists of historical and current trajectories. "
                    f"Historical trajectory contains 40 records, and current trajectory consists of 5 records. "
                    f"Additional Description: "
                    f"Historical trajectory describes travel patterns and frequently visited places, "
                    f"while current trajectories reflect user’s current location and their short-term travel intentions. <|end_prompt|>"
                )
        # 提前tokenize
        tokenized = self.tokenizer(
                [prompt],  # 单样本
                return_tensors="pt"
        )
        with torch.no_grad():
            # 保存token IDs（后续动态生成嵌入）
            self.prompt_token_ids = tokenized.input_ids
            self.prompt_embeddings=self.model.model.get_input_embeddings()(self.prompt_token_ids)
            #self.prompt_embeddings=self.model.get_input_embeddings()(self.prompt_token_ids)
            #(1,prompt_len,llm_dim)
    def __init_persona_prompt(self):
        persona_texts=[
            "Student: This persona represents individuals who typically travel to and from educational institutions at regular times, such as morning arrivals and afternoon departures. Their mobility is highly time-structured and centered around campuses, libraries, and nearby service areas.",
            "Teacher: This persona regularly commutes to educational institutions during weekday mornings and returns home in the late afternoon or early evening. Their travel patterns align closely with school schedules, often involving brief visits to nearby commercial or service areas.",
            "Office Worker: This persona has a fixed daily commute, traveling to office districts or commercial centers in the morning and returning home in the evening. Their mobility follows a consistent weekday routine with limited variation.",
            "Visitor: This persona tends to travel throughout the day with less predictable patterns. They frequently visit tourist attractions, cultural landmarks, dining areas, and shopping districts, especially in central urban zones.",
            "Night Shift Worker: This persona often travels outside of standard business hours, especially during late evenings or at night. Common destinations include hospitals, factories, 24-hour service locations, and late-night dining spots.",
            "Remote Worker: This persona has non-standard travel patterns. They frequently visit coworking spaces, cafés, or quiet public environments at various hours of the day, with flexible scheduling that may shift across weekdays.",
            "Service Industry Worker: This persona has irregular travel times throughout the day. They frequently move between restaurants, shopping areas, entertainment venues, and other customer-facing POIs, reflecting shift-based work in dynamic urban zones.",
            "Public Service Official: This persona often works in rotating shifts, leading to variable travel patterns across different times of the day and night. Common destinations include government offices, transport hubs, hospitals, and administrative centers.",
            "Fitness Enthusiast: his persona is active during early mornings, evenings, or weekends. Their mobility revolves around gyms, sports facilities, parks, and wellness-related POIs. Visit durations tend to be regular and intentional.",
            "Retail Employee: This persona typically begins travel in the late morning and returns in the evening. Their destination patterns focus on malls, retail stores, and service clusters, reflecting the opening and closing hours of retail operations.",
            "Undefined Persona: This persona does not clearly belong to any predefined behavioral category. Their travel patterns may be irregular, spontaneous, or inconsistent across time and location."
        ]
        
        encoded = self.model.tokenizer(
            persona_texts,
            padding='max_length',
            truncation=True,
            max_length=50,
            return_tensors='pt'
        )
        '''
        encoded = self.tokenizer(
            persona_texts,
            padding='max_length',
            truncation=True,
            max_length=50,
            return_tensors='pt'
        )
        '''
        with torch.no_grad():
            self.persona_embeddings=self.model.model.get_input_embeddings()(encoded.input_ids)
            #(11,50,llm_dim)
            self.persona_embeddings = self.persona_embeddings.mean(dim=1)  
            # → (11, 4096)
            #self.persona_embeddings=self.model.get_input_embeddings()(encoded.input_ids)

    def forward(self,batch,accelerator=None):
        
        next_vec,total_entropy=self.predict(batch,accelerator)
        #batch,prompt_seq_len+his_seq_len+cur_seq_len,llm_dim
        #next_loc_emb=next_vec[:,-1,:self.mervec_size]
        #accelerator.print('next_vec',next_vec.shape,next_vec)
        next_loc_emb=next_vec[:,-1,:]
        #batch,dim_llm
        next_dur_emb=next_vec[:,-1,:]
        #batch,dim_for_llm
        next_loc=self.predict_loc(next_loc_emb,accelerator)
        #batch,loc_size

        next_dur=self.predict_dur(next_dur_emb,accelerator)
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        #batch,1

        next_loc_denorm=self.normalize_loc(next_loc,'denorm')
          
        return next_loc,next_loc_denorm,next_dur,total_entropy
        
    def predict(self, batch,accelerator=None):

        history_data_t, current_data_t, history_dur_t, current_dur_t, history_hour_t, \
            current_hour_t, history_day_t, current_day_t = \
            self._load_batch_data(batch, accelerator)
        #batch,his_seq/batch,cur_seq

        history_data_t, current_data_t, history_dur_t, current_dur_t = \
            self._normalize_data(history_data_t, current_data_t, history_dur_t, current_dur_t)

        history_loc_embedding,current_loc_embedding,history_day_embedding,current_day_embedding,\
            history_hour_embedding,current_hour_embedding,history_dur_vector,current_dur_vector=\
            self.__get_embeddings(history_data_t, current_data_t, history_day_t, current_day_t, history_hour_t, current_hour_t, history_dur_t, current_dur_t)

        history_all_embedding,history_hidden_embedding,current_all_embedding=\
            self._get_current_MOE_embedding(history_loc_embedding,current_loc_embedding,history_day_embedding,current_day_embedding,\
                history_hour_embedding,current_hour_embedding,history_dur_vector,current_dur_vector,current_data_t)
        
        dec_out,total_entropy=self.forward_llm(history_all_embedding,history_hidden_embedding,current_all_embedding)
        #batch,prompt_seq_len+his_seq_len+cur_seq_len,llm_dim

        return dec_out,total_entropy

    def _load_batch_data(self,batch,accelerator):    
        history_data_t = self._convert_to_tensor(batch['history_loc'], self.his_seq_len, accelerator)
        current_data_t = self._convert_to_tensor(batch['current_loc'], None, accelerator)

        history_dur_t = self._convert_to_tensor(batch['history_dur'], self.his_seq_len, accelerator)
        current_dur_t = self._convert_to_tensor(batch['current_dur'], None, accelerator)

        history_hour_t = self._convert_to_tensor(batch['history_hour'], self.his_seq_len, accelerator, dtype=torch.long)
        current_hour_t = self._convert_to_tensor(batch['current_hour'], None, accelerator, dtype=torch.long)

        history_day_t = self._convert_to_tensor(batch['history_day'], self.his_seq_len, accelerator, dtype=torch.long)
        current_day_t = self._convert_to_tensor(batch['current_day'], None, accelerator, dtype=torch.long)

        #history_poi_t = self._convert_to_tensor(batch['history_poi'], self.his_seq_len, accelerator)
        #current_poi_t = self._convert_to_tensor(batch['current_poi'], None, accelerator)

        return history_data_t, current_data_t, history_dur_t, current_dur_t, history_hour_t, \
               current_hour_t, history_day_t, current_day_t
    def _convert_to_tensor(self, data, seq_len, accelerator, dtype=torch.float):
        tensor = torch.as_tensor(data).to(dtype).to(accelerator.device)
        if seq_len:
            tensor = tensor[:, -seq_len:]
        return tensor
    def _normalize_data(self, history_data_t, current_data_t, history_dur_t, current_dur_t):   
        history_data_t=self.normalize_loc(history_data_t,'norm')
        current_data_t=self.normalize_loc(current_data_t,'norm')
        #normalize web mercator

        history_dur_t=self.normalize_dur(history_dur_t,'norm')
        current_dur_t=self.normalize_dur(current_dur_t,'norm')
        
        return history_data_t, current_data_t, history_dur_t, current_dur_t
    def __get_embeddings(self, history_data_t, current_data_t, history_day_t, current_day_t,\
                         history_hour_t, current_hour_t, history_dur_t, current_dur_t):
        ################################### initial embedding and vectors##########################################
        history_loc_embedding=self.mer2vec(history_data_t)
        #batch_size,his_seq_len——>batch_size,his_seq_len,self.llm_dim
        current_loc_embedding=self.mer2vec(current_data_t)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,self.llm_dim

        history_day_embedding=self.day_embedding(history_day_t)
        #batch_size,his_seq_len——>batch_size,his_seq_len,self.llm_dim
        current_day_embedding=self.day_embedding(current_day_t)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,self.llm_dim

        history_hour_embedding=self.hour_embedding(history_hour_t)
        #batch_size,his_seq_len——>batch_size,his_seq_len,self.llm_dim
        current_hour_embedding=self.hour_embedding(current_hour_t)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,self.llm_dim

        history_dur_vector=history_dur_t.unsqueeze(-1)
        #batch_size,his_seq_len——>batch_size,his_seq_len,1
        history_dur_vector=self.dur_linear(history_dur_vector)
        #batch_size,his_seq_len,1——>batch_size,his_seq_len,self.llm_dim
        current_dur_vector=current_dur_t.unsqueeze(-1)
        #batch_size,cur_seq_len——>batch_size,cur_seq_len,
        current_dur_vector=self.dur_linear(current_dur_vector)
        #batch_size,cur_seq_len,1——>batch_size,cur_seq_len,self.llm_dim
        #accelerator.print('history_dur_vector',history_dur_vector[52][0])


        return(history_loc_embedding,current_loc_embedding,history_day_embedding,current_day_embedding,\
            history_hour_embedding,current_hour_embedding,history_dur_vector,current_dur_vector)
    def _get_current_MOE_embedding(self,history_loc_embedding,current_loc_embedding,history_day_embedding,current_day_embedding,\
            history_hour_embedding,current_hour_embedding,history_dur_vector,current_dur_vector,current_data_t):
        #######new in NextLocMOE################

        history_all_embedding=torch.concat([history_loc_embedding,
                                            history_day_embedding,
                                            history_hour_embedding,
                                            history_dur_vector],
                                            dim=-1)
        #batch_size,his_seq_len,total_input_dim

        tcn_out = self.tcn(history_all_embedding)  # [B, L, C]
        tcn_out = self.tcn_ln(tcn_out)
        history_hidden_embedding = tcn_out[:, -1, :]

        #batch_size,total_input_dim

        current_base=torch.concat([current_loc_embedding,
                                            current_day_embedding,
                                            current_hour_embedding,
                                            current_dur_vector],
                                            dim=-1)
        #batch_size,cur_seq_len,total_input_dim

        
        routing_input=torch.cat([
            history_hidden_embedding.unsqueeze(1).expand_as(current_base),
            current_base
        ],dim=-1)
        ##batch_size,cur_seq_len,2*total_input_dim


        # 动态获取最新POI权重矩阵
        poi_weights = torch.stack(
            [mlp.weight.T for mlp in self.category_mer2vec],  # (num_poi, 2, mer_dim)
            dim=0
        )  # (num_poi, 2, mer_dim)

        routing_logits=self.poi_emb_prob(routing_input)
        #batch_size,cur_seq_len,len(self.poi_embedding_llm_lst)
        #print('poi_emb_prob',poi_emb_prob.shape)
        routing_weights=torch.softmax(routing_logits,dim=-1)
        top2_values,top2_indices=torch.topk(routing_weights,k=2,dim=-1)
        #batch_size,cur_seq_len,2
        #每个位置使用了哪两个POI的embedding，他们对应的概率是多大
        #print(top2_indices.max())

        selected_weights = poi_weights[top2_indices]
        
        #batch_size,cur_seq_len,2,mer_size,mer_dim
        #每一个#batch_size,cur_seq_len,2，挑self.category_mer2vec_weight对应的num_poi中的一个，将mer_size,mer_dim维度的东西填入
        #每一个batch_size,cur_seq_len的current坐标对应的top1和top2 专家的参数


        coord=current_data_t.unsqueeze(2)
        #batch_size,cur_seq_len,1,2


        projected = torch.einsum(
            'btki,btkij->btkj', 
            coord,  # (batch, cur_seq_len, 1, 2)
            selected_weights  # (batch, cur_seq_len, 2, 2, mer_dim)
        )  
        # 结果形状：(batch, cur_seq_len, 2, mer_dim)
        #对每个轨迹点 (x, y)，用它乘以 top-2 每个专家的 2xD 投影矩阵，输出两个 mer_dim 向量（专家视角下的位置嵌入）。
        '''
            在内部，首先由于corrd的k维度为1，而selected_weights的k维度为2
            PyTorch会自动将corrd扩展为(batch, cur_seq_len, 2, 2)，复制原始数据到新的k维度

            广播后corrd(batch, cur_seq_len, 2, 2)，然后矩阵乘法(2,) × (2,d)——>(d,)

            所以最终结果(batch, cur_seq_len, 2, mer_dim)

            这边每一个(i，j，2,mer_dim)就是batch的第i个轨迹的第j个current record的top2个poi mer embedding
        '''

        weighted = projected * top2_values.unsqueeze(-1)  
        # (batch, cur_seq_len, 2, mer_dim)
        
        summed = weighted.sum(dim=2)
        #(batch, cur_seq_len, mer_dim)

        enhanced_loc = current_loc_embedding + summed
        # general的embedding+top2 POI embedding

        '''
        current_all_embedding=torch.concat([current_loc_embedding,
                                            current_day_embedding,
                                            current_hour_embedding,
                                            current_dur_vector],
                                            dim=-1)
        '''
        current_all_embedding=torch.concat([enhanced_loc,
                                            current_day_embedding,
                                            current_hour_embedding,
                                            current_dur_vector],
                                            dim=-1)
        
        return (history_all_embedding,history_hidden_embedding,current_all_embedding)
    def forward_llm(self,history_all_embedding,history_hidden_embedding,current_all_embedding):
        history_llm_embedding=self.history_llmize_linear(history_all_embedding)
        ##batch_size,his_seq_len,llm_dim

        current_llm_embedding=self.current_llmize_linear(current_all_embedding)
        ##batch_size,cur_seq_len,llm_dim
        total_traj_embedding=torch.concat([history_llm_embedding,current_llm_embedding],dim=1)
        #batch_size,his_seq_len+cur_seq_len,llm_dim
        #这一点存疑，有可能直接prompt+current就可以了
       
        prompt_embedding=self.prompt_embeddings.to(total_traj_embedding.device)
        #(prompt_len,llm_dim)

        # 维度扩展
        batch_size = total_traj_embedding.size(0)
        prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)
        #(batch,prompt_len,llm_dim)

        # 最终拼接
        total_embedding = torch.cat([prompt_embedding, total_traj_embedding], dim=1)
        #(batch,prompt_len+his_seq_len+cur_seq_len,llm_dim)

        #total_embedding=total_traj_embedding
        
        #total_embedding=total_traj_embedding
        #accelerator.print('total_embedding',total_embedding.shape,total_embedding[52][0])
        persona_embeddings=self.persona_embeddings.to(total_traj_embedding.device)
        
        dec_out, entropy_losses = self.model(total_embedding,
            None,
            history_hidden_embedding,
            persona_embeddings)
            
        #dec_out=self.model(inputs_embeds=total_embedding).last_hidden_state
        #batch,prompt_seq_len+his_seq_len+cur_seq_len,llm_dim

        total_entropy = sum(entropy_losses) / len(entropy_losses)

        return dec_out,total_entropy
        #return dec_out, torch.tensor(0.0, device=dec_out.device)



    def predict_loc(self,next_llm_emb,accelerator):
        
        next_loc=self.output_loc(next_llm_emb)
        #batch,loc_size
        #accelerator.print('next_loc before tanh',next_loc.shape,next_loc[0])
        #accelerator.next_loc=torch.tanh(next_loc)
        #这个tanh效果其实不大
        #accelerator.print('next_loc after tanh',next_loc.shape,next_loc[0])
        return next_loc
        #batch,2

    def predict_dur(self,next_llm_emb,accelerator):
        
        next_dur=self.output_dur(next_llm_emb)
        #batch,1
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        next_dur=torch.relu(next_dur)
        #accelerator.print('next_dur',next_dur.shape,next_dur)
        return next_dur

    def calculate_loss(self,batch,accelerator):

        
        next_loc,next_loc_denorm,next_dur,total_entropy=self.forward(batch,accelerator)
                
        next_dur_denorm=self.normalize_dur(next_dur,'denorm')
        
        true_loc=torch.as_tensor(batch['target']).float().to(accelerator.device)

        #accelerator.print("Pred loc:", next_loc_denorm[0])
        #accelerator.print("True loc:", true_loc[0])
        
        distance_loss=torch.linalg.norm(next_loc_denorm- true_loc, dim=1).mean()
        # 预测位置和真实位置之间的平均欧几里得距离
        #accelerator.print(distance_loss)
        

        total_loss=distance_loss+self.config['lambda_']*total_entropy
        return total_loss,distance_loss,self.config['lambda_']*total_entropy

        