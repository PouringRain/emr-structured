from collections import OrderedDict

def data_process(data, max_len):
    mapping = OrderedDict()  # 存文本与片段的映射列表
    id = 0  # 记录第几个片段
    sents = []
    code_mapping = OrderedDict()
    for i, d in enumerate(data):
        code = d['code']
        text = d['text']
        lenth = len(text)
        code_mapping[i] = code
        if len(text) <= max_len:
            mapping[i] = [id]
            id += 1
            sents.append(text)
            continue

        s_idx, e_idx = 0, max_len
        while True:
            sec = text[s_idx: e_idx]
            if i not in mapping:
                mapping[i] = [id]
            else:
                mapping[i].append(id)
            id += 1
            sents.append(sec)

            if e_idx >= lenth: break
            s_idx = e_idx
            e_idx = min(lenth, e_idx+max_len)

    return sents, mapping, code_mapping

def post_process(res, mapping, code_mapping, max_len):

    entities_list = []
    for k, v in mapping.items():
        cur = []
        start, end = v[0], v[-1]
        for i, entities in enumerate(res[start: end+1]):
            # 处理下标
            for entity in entities:
                entity['start_idx'] += max_len * i
                entity['end_idx'] += max_len * i
            cur += entities
        entities_list.append({
            "code": code_mapping[k],
            "entities": cur
        })

    return entities_list


if __name__=='__main__':
    data = [
                {
                    "text": "否认肝炎史、疟疾史、结核史，否认高血压史、冠心病史，否认糖尿病史、脑血管病史、精神病史，4年前因扁桃体肥大行“扁桃体切除术”8年前行胆囊切除术。否认外伤史、输血史，否认过敏史，预防接种史不详。",
                    "code": "history_of_past_illness"
                },
                {
                    "text": "患者10余年前无明显诱因出现左膝关节疼痛、活动受限，行走过久及劳累时疼痛明显，休息后缓解，偶有关节交锁及打软腿，无腰痛及双下肢麻木。疼痛逐渐加重。左下肢逐渐内翻畸形。2年前自觉疼痛明显加重。就诊于当地医院，予以针灸、口服药物、中药外敷等等治疗，效果不佳。为求进一步治疗，就诊于我院，门诊以\"左膝骨性关节炎\"收入我科。患者近期，精神食欲睡眠可，大小便可。",
                    "code": "history_of_past_illness"
                }
            ]
    sents, mapping, code_mapping = data_process(data, 50)
    print(sents)
    print(mapping)
    print(code_mapping)
