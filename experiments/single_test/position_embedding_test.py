# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: position_embedding_test
    Author: czh
    Create Date: 2021/8/6
--------------------------------------
    Change Activity: 
======================================
"""
from transformers import BertTokenizerFast
from nlp.models.utils import _generate_relative_positions_embeddings

bert_model_name_or_path = "/data/chenzhihao/chinese-roberta-ext"
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name_or_path)
text = ["俄罗斯卫星网刚刚消息称，美军在喀布尔机场向阿富汗平民开火，已致数人死亡。"]
ids = tokenizer.batch_encode_plus(text, return_tensors='pt', max_length=128, padding="max_length")
# print(ids)
input_ids = ids["input_ids"]
token_type_ids = ids["token_type_ids"]
print(token_type_ids)

# embedding = PositionEmbedding(128, 768, merge_mode='zero', hierarchical=True, embeddings_initializer='xavier_uniform')
# embedding = SinusoidalPositionEmbedding(output_dim=768, merge_mode='zero')
# embedding = RoFormerSinusoidalPositionalEmbedding(128, 768)
# embedding = RelativePositionEmbedding(128*2+1, 768)
# embedding = RelativePositionEmbeddingT5(input_dim=128*2+1, output_dim=768)
# pos = embedding(input_ids)

# lm = LM_Mask()
# pos = lm.lm_mask(64, 64)

# ulm = UniLM_Mask()
# pos = ulm.unilm_mask(token_type_ids, 128-token_type_ids.size(1))

pos = _generate_relative_positions_embeddings(seq_length=128, embed_dim=64, max_relative_position=128)
print(pos)
print(pos.size())

