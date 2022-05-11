#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request
import json
import traceback
from predict import batch_predict
from utils.processor import data_process, post_process
from config import config
import logging

app = Flask(__name__)

@app.route('/health', methods=['POST'])
def health_check():
    return "200 OK"

@app.route('/ner', methods=['POST'])
def ner():
    final_result = dict()
    try:
        data = json.loads(request.data)  # 读取数据
    except Exception as e:
        final_result['data'] = []
        final_result['status'] = "Unavailable"
        final_result['message'] = "请检查数据输入格式！{}".format(traceback.format_exc())
        return json.dumps(final_result, ensure_ascii=False)
    try:
        # 判断为空
        if len(data)==0:
            final_result['data'] = []
            final_result['status'] = "SingleFail"
            final_result['message'] = "请求列表为空！"
            return json.dumps(final_result, ensure_ascii=False)

        sents, mapping, code_mapping = data_process(data, config.MAX_LEN)
        result_info = batch_predict(sents)
        result_info = post_process(result_info, mapping, code_mapping, config.MAX_LEN)
        final_result['data'] = result_info
        final_result['status'] = "Succeed"
        final_result['message'] = "成功返回"
        return json.dumps(final_result, ensure_ascii=False)

    except Exception as e:
        final_result['data'] = []
        final_result['status'] = "SingleFail"
        final_result['message'] = traceback.format_exc()
        logging.error("request get some trouble: %s\n%s" % (e, traceback.format_exc()))
        return json.dumps(final_result, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=False)
