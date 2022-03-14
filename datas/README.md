# 数据格式
## 命名实体识别
### Cluener
数据格式如下（每行一个json。若为无实体样本，可不传入label字段）：
```
{"text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，", "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}}
{"text": "温格的球队终于又踢了一场经典的比赛，2比1战胜曼联之后枪手仍然留在了夺冠集团之内，", "label": {"organization": {"曼联": [[23, 24]]}, "name": {"温格": [[0, 1]]}}}
{"text": "突袭黑暗雅典娜》中Riddick发现之前抓住他的赏金猎人Johns，", "label": {"game": {"突袭黑暗雅典娜》": [[0, 7]]}, "name": {"Riddick": [[9, 15]], "Johns": [[28, 32]]}}}
```

## 事件抽取
### DuEE1.0
```
{"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
{"text": "前两天，被称为 “ 仅次于苹果的软件服务商 ” 的 Oracle（ 甲骨文 ）公司突然宣布在中国裁员。。", "id": "b90900665eee74f658eed125a321ee06", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 48, "arguments": [{"argument_start_index": 0, "role": "时间", "argument": "前两天", "alias": []}, {"argument_start_index": 4, "role": "裁员方", "argument": "被称为 “ 仅次于苹果的软件服务商 ” 的 Oracle（ 甲骨文 ）公司", "alias": []}], "class": "组织关系"}]}
```
event_schema.json
```
{"event_type": "财经/交易-出售/收购", "role_list": [{"role": "时间"}, {"role": "出售方"}, {"role": "交易物"}, {"role": "出售价格"}, {"role": "收购方"}], "id": "804336473abe8b8124d00876a5387151", "class": "财经/交易"}
{"event_type": "财经/交易-跌停", "role_list": [{"role": "时间"}, {"role": "跌停股票"}], "id": "29a8f7417bf8867ddb8521f647a828d8", "class": "财经/交易"}
```
### tplinker
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
## 关系抽取
### kg2019
百度知识抽取竞赛，千言数据
```
{"postag": [{"word": "查尔斯", "pos": "nr"}, {"word": "·", "pos": "w"}, {"word": "阿兰基斯", "pos": "nr"}, {"word": "（", "pos": "w"}, {"word": "Charles Aránguiz", "pos": "nz"}, {"word": "）", "pos": "w"}, {"word": "，", "pos": "w"}, {"word": "1989年4月17日", "pos": "t"}, {"word": "出生", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "智利圣地亚哥", "pos": "ns"}, {"word": "，", "pos": "w"}, {"word": "智利", "pos": "ns"}, {"word": "职业", "pos": "n"}, {"word": "足球", "pos": "n"}, {"word": "运动员", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "司职", "pos": "v"}, {"word": "中场", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "效力", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "德国", "pos": "ns"}, {"word": "足球", "pos": "n"}, {"word": "甲级", "pos": "a"}, {"word": "联赛", "pos": "n"}, {"word": "勒沃库森足球俱乐部", "pos": "nt"}], "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", "spo_list": [{"predicate": "出生地", "object_type": "地点", "subject_type": "人物", "object": "圣地亚哥", "subject": "查尔斯·阿兰基斯"}, {"predicate": "出生日期", "object_type": "Date", "subject_type": "人物", "object": "1989年4月17日", "subject": "查尔斯·阿兰基斯"}]}
{"postag": [{"word": "《", "pos": "w"}, {"word": "离开", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "是", "pos": "v"}, {"word": "由", "pos": "p"}, {"word": "张宇", "pos": "nr"}, {"word": "谱曲", "pos": "v"}, {"word": "，", "pos": "w"}, {"word": "演唱", "pos": "v"}], "text": "《离开》是由张宇谱曲，演唱", "spo_list": [{"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "张宇", "subject": "离开"}, {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张宇", "subject": "离开"}]}
```
