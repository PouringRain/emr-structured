# -*- coding: utf-8 -*-

from transformers import BertModel, BertTokenizerFast
from models.GlobalPointer import EffiGlobalPointer as GlobalPointer
import json
import torch
import numpy as np
from tqdm import  tqdm

bert_model_path = r'D:\BaiduNetdiskDownload\bert\torch_roberta_wwm' #your RoBert_large path

save_model_path = './outputs/TEST_EP_L5.pth' #67.94%
device = torch.device("cuda:0")

max_len = 256
ent2id, id2ent = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder = BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, 9 , 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()

def NER_RELATION(text, tokenizer, max_len=max_len):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
    scores = model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l]})

    return {"text": text, "entities":entities}

if __name__ == '__main__':
    all_ = []
    for d in tqdm(json.load(open('datasets/CMeEE_test.json', encoding='utf-8'))):
        all_.append(NER_RELATION(d["text"], tokenizer=tokenizer))
    json.dump(
        all_,
        open('./outputs/CMeEE_test.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )