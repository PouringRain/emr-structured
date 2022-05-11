def fixup(data):
    '''
    目前的问题为：
    1. 去重
    2. 合并相连的时间 如抽取2021年，5月，1日，连起来为2021年5月1日
    :param :
    :return:
    '''

    text, entities = data['text'], data['entities']
    # 去重
    info = []
    for x in entities:
        if x not in info: info.append(x)

    if len(info) <= 1:
        data = {
            'text': text,
            'entities': info
        }
        return data

    ret = [info[0]]
    # 合并挨着的实体
    info.sort(key=lambda x: x['start'])
    for x in info:
        if ret[-1]['end'] >= x['start']:
            start = ret[-1]['start']
            end = max(ret[-1]['end'], x['end'])
            ret[-1]['end'] = end
            ret[-1]['word'] = text[start: end]
        else:
            ret.append(x)

    # 为了和实体识别结果保持统一，end_span需要 -1
    # 时间标准化也在这里做吧
    for x in ret: x['end'] -= 1

    data = {
        'text': text,
        'entities': ret
    }
    return data


def combine_entities_time(text, entities, parser):
    ret = parser.match_all(text)
    ret = fixup(ret)

    for r in ret['entities']:
        entities.append({
            "start_idx": r['start'],
            "end_idx": r['end'],
            "type": "time",
            "entity": r['word'],
        })

    return entities