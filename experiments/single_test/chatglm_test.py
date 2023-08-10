#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--------------------------------
Author：czh
date：2023/3/24
--------------------------------
"""
from transformers import AutoModel, AutoTokenizer

device = "cuda:3"
model_path = "/root/work2/work2/chenzhihao/pretrained_models/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True, revision=True).half().to('mps')
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, revision=True).half().to(device)
model = model.eval()

history = []
while True:
    query = input("\nuser(q to stop): ")
    if query.strip() == 'q':
        break

    response, history = model.chat(tokenizer, query.strip(), history=history)
    print("\nresponse: ", response)
