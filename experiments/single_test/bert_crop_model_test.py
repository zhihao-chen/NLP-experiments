# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: bert_crop_model_test
    Author: czh
    Create Date: 2021/8/10
--------------------------------------
    Change Activity: 
======================================
"""
from transformers import BertConfig, BertPreTrainedModel, BertModel
from nlp.models.bertcrop import BertCropModel, save_specify_num_hidden_layers_state


bert_model_path = "/Users/czh/Downloads/chinese_bert_wwm"
bert_config = BertConfig.from_json_file(bert_model_path+'/config.json')
bert_config.num_hidden_layers = 1

# state_dict = torch.load(bert_model_path+'/pytorch_model.bin')
# bert_model.init_from_pretrained(state_dict)


class MyModel(BertPreTrainedModel):
    def __init__(self, config, pretrained_bert_path):
        super(MyModel, self).__init__(config)

        self.bert = BertCropModel(config)
        # state_dict = torch.load(pretrained_bert_path+'/pytorch_model.bin')
        # init_from_pretrained(self.bert, state_dict, True)


bert_model_ = BertModel.from_pretrained(bert_model_path)
save_specify_num_hidden_layers_state(bert_model_, [1], "./pytorch_model_0_layer.bin")

bert_model = MyModel.from_pretrained("./pytorch_model_0_layer.bin", config=bert_config, pretrained_bert_path=bert_model_path)

for n, p in bert_model.named_parameters():
    print(n)
