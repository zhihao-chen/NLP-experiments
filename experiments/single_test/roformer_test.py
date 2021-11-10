# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: roformer_test
    Author: czh
    Create Date: 2021/9/3
--------------------------------------
    Change Activity: 
======================================
"""
import torch
from transformers import RoFormerModel, RoFormerForMaskedLM, RoFormerTokenizer


text = "时光流逝，但有些日子注定被永久铭记。在抗战胜利76周年纪念日到来之际，各地举办了形式多样的活动，让人们在回望历史中收获心灵的洗礼、得到思想的升华。河南南阳市当地媒体推出“纪念抗战胜利76周年 山河同在”系列报道，与读者一起回望气壮山河的抗日史诗，凝聚兴我中华的磅礴伟力；甘肃永昌县中小学的“开学第一课”以牢记历史为切入点，通过观看抗战专题视频、抗战知识问答等形式，回顾中国人民艰苦抗战的峥嵘岁月；广西桂林市举办纪念抗战胜利76周年文艺演出活动,百余位党员群众追忆革命先辈的艰辛历程，讴歌永远跟党走的坚定誓言。历史是最好的教科书，也是最好的清醒剂。从1931年日本军国主义的铁蹄蹂躏中国东北的白山黑水，到1945年9月2日，日本代表在无条件投降书上签字，十四年抗战的血与火背后，是3500多万同胞伤亡，930余座城市先后被占，4200万难民无家可归。于民族危难之际，中国共产党支撑起救亡图存的希望。从打破“日军不可战胜”神话的平型关大捷，到粉碎侵略者“囚笼政策”的百团大战；从让日军“名将之花”凋谢在太行山上的黄土岭战斗，到打响华中反攻第一枪的车桥战役，——在中国共产党的领导下，无数不甘屈辱的中华儿女前赴后继，以血肉之躯筑起新的长城，赢得了自1840年鸦片战争以来抗击外敌入侵的第一次完全胜利！为争取世界和平的伟大事业，作出了永载史册的重大贡献！战争硝烟早已散去，苦难岁月还需铭记，并非是要背着包袱前行，而是只有牢记来时的路，才能走向更远的前方。正如联合国的呼吁：“我们有责任见证苦难永远不再重演，受难者的记忆被永久尊重。”我们永远不会忘记，“名将以身殉国家，愿拼热血卫吾华”的左权，“未惜头颅新故国，甘将热血沃中华”的赵一曼，弹尽后毅然投江的八名抗联女兵，打完最后一粒子弹后壮烈跳崖的狼牙山五壮士……岁月长河，历史足迹不容磨灭；时代变迁，英雄精神熠熠发光。当76年前的历史场景在这一天再次重现，当战争创伤在和平年代只能靠记忆的方式还原，每一个中华儿女都已然在心中默默地葆有一份肃穆与庄重。"
model_name = "junnyu/roformer_chinese_base"
tokenizer = RoFormerTokenizer.from_pretrained(model_name)
pt_model = RoFormerForMaskedLM.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    pt_outputs = pt_model(**input_ids).logits[0]
pt_outputs_sentence = "pytorch: "
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        tokens = tokenizer.convert_ids_to_tokens(pt_outputs[i].topk(k=5)[1])
        pt_outputs_sentence += "[" + "||".join(tokens) + "]"
    else:
        pt_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True))
print(pt_outputs_sentence)

