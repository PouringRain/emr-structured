import re
import logging
import traceback
from collections import namedtuple
from utils.time_post_process import fixup

class Regexparser():
    def __init__(self, regex, logger=None):
        '''
        正则匹配的构造函数
        :param regex: 加载正则，list形式, 用于正则匹配的结构为namedtuple
        :param logger: 日志logger
        '''
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger('utils.time.parser')
        Regex = namedtuple("Regex", ["pattern"])
        self.re = []
        for t in regex:
            try:
                pattern = re.compile(t)
            except Exception as e:
                self.logger.error("regex has some problem! %s\n%s" % (e, traceback.format_exc()))
                continue
            self.re.append(Regex(pattern=pattern))
        self.logger.info('正则parser模块初始化完毕！')

    def match_all(self, content):
        '''
        用批量正则匹配文本
        :param content: 待匹配文本
        :return: 匹配到的内容和位置，list形式返回
        '''
        info = []
        for i in self.re:
            result_finditer = i.pattern.finditer(content)
            for k in result_finditer:
                span = k.span()
                info.append({'word': k.group(), 'start': span[0], 'end': span[1]})

        ret = {
            'text': content,
            'entities': info
        }
        return ret

if __name__=='__main__':
    regex = []
    with open(r'C:\Users\jshen\PycharmProjects\emr_info_extraction\config\re', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            regex.append(line.strip())

    parser = Regexparser(regex=regex)
    sentence = r'慢性胃炎病史2年余，时有反酸、烧心感，平素口服“气滞胃痛颗粒、胃苏颗粒”改善症状。【体质情况】体质情况既往体质良好，【疾病外伤史情况】疾病外伤史情况否认“高血压、糖尿病、慢性肾病、冠心病、慢性肝病、慢性阻塞性肺疾病”等病史，否认外伤史，【传染病史情况】传染病史情况否认“肝炎、结核”等病史，【手术史外伤史情况】手术史外伤史情况否认手术史，【输血史情况】输血史情况否认输血史及血液制品使用史，【过敏史预防史情况】过敏史预防史情况对青霉素类药物过敏，否认其他药物过敏史,预防接种史不详。'
    print(fixup(parser.match_all(sentence)))