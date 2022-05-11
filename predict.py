# -*- coding: utf-8 -*-
from transformers import BertModel, BertTokenizerFast
from models.GlobalPointer import EffiGlobalPointer as GlobalPointer
import torch
import numpy as np
from utils.data_loader import EntDataset, load_test_data
from torch.utils.data import DataLoader
from config import config
from utils.neg_match import neg_match
from utils.parser import Regexparser
from utils.time_post_process import combine_entities_time

bert_model_path = config.BERT_MODEL_PATH
save_model_path = config.SAVE_MODEL_PATH
device = torch.device("cuda:0")

max_len = config.MAX_LEN
ent2id, id2ent = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder = BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, 9, 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()

regex = []
with open(config.RE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        regex.append(line.strip())

parser = Regexparser(regex=regex)

def batch_predict(data):
    pred = []
    ner_evl = EntDataset(load_test_data(data), tokenizer=tokenizer)
    ner_loader_evl = DataLoader(ner_evl, batch_size=len(data), collate_fn=ner_evl.collate, shuffle=False, num_workers=0)
    for batch in ner_loader_evl:
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device)
        y_pred = model(input_ids, attention_mask, segment_ids)
        y_pred = y_pred.data.cpu().numpy()

        for k, b in enumerate(y_pred):
            new_span, entities = [], []
            token2char_span_mapping = tokenizer(raw_text_list[k], return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])

            for l, start, end in zip(*np.where(b > 0)):
                start_idx, end_idx = new_span[start][0], new_span[end][-1]
                entities.append({"start_idx": start_idx,
                                 "end_idx": end_idx,
                                 "type": id2ent[l],
                                 "entity": raw_text_list[k][start_idx: end_idx+1]
                                 })
            entities = neg_match(config.negword_list, entities, raw_text_list[k])
            entities = combine_entities_time(raw_text_list[k],entities, parser)
            pred.append(entities)

    return pred


if __name__=='__main__':
    data = ['否认肝炎史、疟疾史、结核史，否认高血压史、冠心病史，否认糖尿病史、脑血管病史、精神病史，4年前因扁桃体肥大行“扁桃体切除术”8年前行胆囊切除术。否认外伤史、输血史，否认过敏史，预防接种史不详。',
            '者10余年前无明显诱因出现左膝关节疼痛、活动受限，行走过久及劳累时疼痛明显，休息后缓解，偶有关节交锁及打软腿，无腰痛及双下肢麻木。疼痛逐渐加重。左下肢逐渐内翻畸形。2年前自觉疼痛明显加重。就诊于当地医院，予以针灸、口服药物、中药外敷等等治疗，效果不佳。'
            ]
    res = batch_predict(data)
    print(res)