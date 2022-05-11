# -*- coding: utf-8 -*-
from utils.data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
from models.GlobalPointer import EffiGlobalPointer, MetricsCalculator
from tqdm import tqdm
from utils.logger import logger
from utils.bert_optimization import BertAdam
bert_model_path = r'D:\BaiduNetdiskDownload\bert\torch_roberta_wwm' #RoBert_large 路径
train_cme_path = './datasets/CMeEE_train.json' #CMeEE 训练集
eval_cme_path = './datasets/CMeEE_dev.json' #CMeEE 测试集
device = torch.device("cuda:0")

BATCH_SIZE = 12
ENT_CLS_NUM = 9

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

#tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

#train_data and val_data
ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train, batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=0)
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl, batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=0)

#GP MODEL
encoder = BertModel.from_pretrained(bert_model_path)
model = EffiGlobalPointer(encoder, ENT_CLS_NUM, 64).to(device) # 9个实体类型

#attack
fgm = FGM(model)

#optimizer
def set_optimizer( model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer

EPOCH = 10
optimizer = set_optimizer(model, train_steps=(int(len(ner_train) / BATCH_SIZE) + 1) * EPOCH)
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss

metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0

def train():
    for eo in range(EPOCH):
        total_loss, total_f1 = 0., 0.
        for idx, batch in enumerate(ner_loader_train):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # 对抗训练
            fgm.attack()
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, segment_ids)
            loss = loss_fun(logits, labels)
            loss.backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            sample_f1 = metrics.get_sample_f1(logits, labels)
            total_loss += loss.item()
            total_f1 += sample_f1.item()

            avg_loss = total_loss / (idx + 1)
            avg_f1 = total_f1 / (idx + 1)
            if idx % 10 == 0:
                logger.info("trian_loss:%f\t train_f1:%f"%(avg_loss, avg_f1))

        with torch.no_grad():
            total_f1_, total_precision_, total_recall_ = 0., 0., 0.
            model.eval()
            for batch in tqdm(ner_loader_evl, desc="Valing"):
                raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
                input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                    device), segment_ids.to(device), labels.to(device)
                logits = model(input_ids, attention_mask, segment_ids)
                f1, p, r = metrics.get_evaluate_fpr(logits, labels)
                total_f1_ += f1
                total_precision_ += p
                total_recall_ += r
            avg_f1 = total_f1_ / (len(ner_loader_evl))
            avg_precision = total_precision_ / (len(ner_loader_evl))
            avg_recall = total_recall_ / (len(ner_loader_evl))
            # logger.info("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1,avg_precision,avg_recall))
            logger.info("EPOCH:%d\t EVAL_F1:%f\tPrecision:%f\tRecall:%f\t"%(eo, avg_f1,avg_precision,avg_recall))
            # if avg_f1 > max_f:
            torch.save(model.state_dict(), './outputs/TEST_EP_L{}.pth'.format(eo))
                # max_f = avg_f1
            model.train()

if __name__ == '__main__':
    train()
