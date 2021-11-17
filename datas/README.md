#数据格式
##命名实体识别
###Cluener
数据格式如下（每行一个json。若为无实体样本，可不传入label字段）：
```
{"text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，", "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}}
{"text": "温格的球队终于又踢了一场经典的比赛，2比1战胜曼联之后枪手仍然留在了夺冠集团之内，", "label": {"organization": {"曼联": [[23, 24]]}, "name": {"温格": [[0, 1]]}}}
{"text": "突袭黑暗雅典娜》中Riddick发现之前抓住他的赏金猎人Johns，", "label": {"game": {"突袭黑暗雅典娜》": [[0, 7]]}, "name": {"Riddick": [[9, 15]], "Johns": [[28, 32]]}}}
```

##事件抽取
###DuEE1.0
```
{"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
{"text": "前两天，被称为 “ 仅次于苹果的软件服务商 ” 的 Oracle（ 甲骨文 ）公司突然宣布在中国裁员。。", "id": "b90900665eee74f658eed125a321ee06", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 48, "arguments": [{"argument_start_index": 0, "role": "时间", "argument": "前两天", "alias": []}, {"argument_start_index": 4, "role": "裁员方", "argument": "被称为 “ 仅次于苹果的软件服务商 ” 的 Oracle（ 甲骨文 ）公司", "alias": []}], "class": "组织关系"}]}
```
event_schema.json
```
{"event_type": "财经/交易-出售/收购", "role_list": [{"role": "时间"}, {"role": "出售方"}, {"role": "交易物"}, {"role": "出售价格"}, {"role": "收购方"}], "id": "804336473abe8b8124d00876a5387151", "class": "财经/交易"}
{"event_type": "财经/交易-跌停", "role_list": [{"role": "时间"}, {"role": "跌停股票"}], "id": "29a8f7417bf8867ddb8521f647a828d8", "class": "财经/交易"}
```
###tplinker
```
{"text": "", 
"id": "valid_0",
"event_list":[
        {
            "event_type": "<event type>",
            "trigger": "<trigger text>",
            "trigger_char_span": "<the character level offset of the trigger>",
            "argument_list":[
                {
                    "text": "<argument text>",
                    "type": "<argument role>",
                    "char_span": "<the character level offset of this argument>, [13, 15] or [[13, 15], [26, 29]]"
                }
            ]
        }
    ]} 
```
