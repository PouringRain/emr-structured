import re

def neg_match(negword_list, entities, text):
    '''
    negword_list: 存放否定词
    illness: 疾病

    return: 若存在该疾病则返回True，否则返回False
    '''
    for r in entities:
        # 拆短句
        short_sentences = re.split(r'[,，:：;；。]', text)
        if r['type'] in ['dis', 'sym']:
            flag = True
            illness = r['entity']
            for short_sentence in short_sentences:
                for negword in negword_list:
                    if negword in short_sentence and illness in short_sentence:

                        r['is_neg'] = 'neg'
                        flag = False
            if flag:
                r['is_neg'] = 'pos'

    return entities