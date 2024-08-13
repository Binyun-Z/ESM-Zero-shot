import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import bootstrap
import numpy as np
import re
import os
import shutil
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
class Mutation_Set(Dataset):
    def __init__(self, data, tokenizer, sep_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = sep_len
        self.seq, self.attention_mask = tokenizer(list(self.data['mutated_sequence']), padding='max_length',
                                                  truncation=True,
                                                  max_length=self.seq_len).values()

        target = list(data['target_seq'])
        self.target, self.tgt_mask = tokenizer(target, padding='max_length', truncation=True,
                                               max_length=self.seq_len).values()
        self.score = torch.tensor(np.array(self.data['DMS_score']))
        self.pid = np.asarray(data['PID'])

        if type(list(self.data['mut_pos'])[0]) != str:
            self.position = [[u] for u in self.data['mut_pos']]

        else:
            self.position = []
            for u in self.data['mut_pos']:
                p = re.findall(r'\d+', u)
                pos = [int(v) for v in p]
                self.position.append(pos)



    def __getitem__(self, idx):
        return [self.seq[idx], self.attention_mask[idx], self.target[idx],self.tgt_mask[idx] ,self.position[idx], self.score[idx], self.pid[idx]]

    def __len__(self):
        return len(self.score)

    def collate_fn(self, data):
        seq = torch.tensor(np.array([u[0] for u in data]))
        att_mask = torch.tensor(np.array([u[1] for u in data]))
        tgt = torch.tensor(np.array([u[2] for u in data]))
        tgt_mask = torch.tensor(np.array([u[3] for u in data]))
        pos = [torch.tensor(u[4]) for u in data]
        score = torch.tensor(np.array([u[5] for u in data]), dtype=torch.float32)
        pid = torch.tensor(np.array([u[6] for u in data]))
        return seq, att_mask, tgt, tgt_mask, pos, score, pid
        
def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]

def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    sr = (sr,)
    ci = list(bootstrap(sr, np.mean).confidence_interval)
    return mean, std, ci

def get_pos(row):
    pos = []
    for mut in row['mutant'].split(':'):
        result = int(re.findall(r'\d+', mut)[0])
        pos.append(result)
    if len(pos)<=1:return pos[0]
    else:
        return pos
def get_wt(seq, mut):
    # mut的输入为A2D, or A2D:B3C
    pos = []
    chars = []
    
    for mutation in mut.split(':'):
        original_char = mutation[0]  # 获取原始字符
        position = int(re.findall(r'\d+', mutation)[0])  # 获取位置
        pos.append(position)
        chars.append(original_char)  # 保存原始字符
    
    seq = list(seq)
    for i, p in enumerate(pos):
        seq[p - 1] = chars[i]  # 替换为原始字符
    return ''.join(seq)

def compute_score(model, seq, mask, wt, pos, tokenizer):
    '''
    compute mutational proxy using masked marginal probability
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    seq = seq.to('cuda') 
    mask = mask.to('cuda') 
    wt = wt.to('cuda') 
    model = model.to('cuda')
    device = seq.device
    model.eval()

    mask_seq = seq.clone()
    m_id = tokenizer.mask_token_id

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos] = m_id

    out = model(mask_seq, mask, output_hidden_states=True)
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)

    for i in range(batch_size):

        mut_pos = pos[i]
        score_i = log_probs[i]
        wt_i = wt[i]
        seq_i = seq[i]
        scores[i] = torch.sum(score_i[mut_pos, seq_i[mut_pos]])-torch.sum(score_i[mut_pos, wt_i[mut_pos]])

    return scores, logits

if __name__ == '__main__':
    basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    data_path = './substitutions_singles'

    # file_list = [file for  file in  os.listdir(data_path) if file.endswith('.csv')]

    file_list=[
        "BLAT_ECOLX_Jacquier_2013.csv",
        "CALM1_HUMAN_Weile_2017.csv",
        "DYR_ECOLI_Thompson_2019.csv",
        "DLG4_RAT_McLaughlin_2012.csv",
        "REV_HV1H2_Fernandes_2016.csv",
        "TAT_HV1BR_Fernandes_2016.csv",
        "RL40A_YEAST_Roscoe_2013.csv",
        "P53_HUMAN_Giacomelli_2018_WT_Nutlin.csv"
    ]
    sr_list = []
    for csv_file in file_list:
        df = pd.read_csv(os.path.join(data_path,csv_file))
        df['mut_pos'] = df.apply(get_pos,axis = 1)
        wt_seq = get_wt(df['mutated_sequence'][0],df['mutant'][0])
        df['target_seq'] = wt_seq
        df['PID'] = df.index
        df = df[df['mut_pos']<1023]
        dfset = Mutation_Set(data=df,tokenizer=tokenizer)

        dfloader = DataLoader(dfset, batch_size=1, collate_fn=dfset.collate_fn, shuffle=True,num_workers=96)

        basemodel.eval()
        seq_list = []
        score_list = []
        gscore_list = []
        with torch.no_grad():
            for step, data in enumerate(dfloader):
                seq, mask = data[0], data[1]
                wt, wt_mask = data[2], data[3]
                pos = data[4]
                golden_score = data[5]
                pid = data[6]

                score, logits = compute_score(basemodel, seq, mask, wt, pos, tokenizer)

                score = score.cuda()


                score = np.asarray(score.cpu())
                golden_score = np.asarray(golden_score.cpu())
                score_list.extend(score)
                gscore_list.extend(golden_score)
        score_list = np.asarray(score_list)
        gscore_list = np.asarray(gscore_list)
        sr = spearman(score_list, gscore_list)
        sr_list.append(sr)
        print(f'dataset------------{csv_file}-----------------spearman-------{sr}')
    pd.DataFrame({'dataset':file_list,'spearman':sr_list}).to_csv('./selected_data_esm2_spearman.csv',index = False)

