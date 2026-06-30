"""生成一个小型合成语料，用于教学验证整条流水线。

特点:
    - 完全合成，无版权问题
    - 句式有限但有内部模式，nano 模型能学到统计规律
    - 同时生成对应的 SFT (Q&A) 数据

用法:
    python scripts/make_sample_data.py
"""
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

NAMES = ["小明", "小红", "小李", "阿强", "丽丽", "张三", "李四", "王五", "晓东", "美美"]
ANIMALS = ["小猫", "小狗", "小鸟", "兔子", "金鱼", "乌龟", "鹦鹉", "仓鼠"]
PLACES = ["公园", "学校", "图书馆", "海边", "山上", "商店", "市场", "操场", "森林", "湖边"]
ACTIVITIES = [
    "看书", "跑步", "画画", "唱歌", "下棋", "种树", "钓鱼", "做饭",
    "写字", "打球", "散步", "听音乐", "拍照", "讲故事",
]
WEATHERS = ["晴朗", "多云", "下雨", "刮风", "下雪", "阴天"]
FEELINGS = ["开心", "兴奋", "安静", "认真", "好奇", "满足", "惊讶", "感动"]
TIMES = ["早上", "中午", "下午", "晚上", "昨天", "今天", "前天"]


STORY_TEMPLATES = [
    "{time}天气{weather}，{name}去了{place}。{name}在那里{act}，感觉非常{feel}。",
    "{name}很喜欢{animal}。{time}，{name}带着{animal}一起去{place}{act}。",
    "{time}{name}遇到了{name2}。他们一起去{place}{act}，玩得很{feel}。",
    "在{place}里，{name}看见一只{animal}。{name}觉得很{feel}，决定回家告诉{name2}。",
    "{name}最喜欢的事情是{act}。{time}，{name}又去{place}{act}了。",
    "{time}的{place}非常{feel}。{name}和{name2}坐在那里，看着天空慢慢变暗。",
    "{name}告诉{name2}：今天我在{place}{act}了，天气很{weather}。",
    "{animal}是{name}最好的朋友。每天{time}，{name}都会带{animal}去{place}。",
]

QA_TEMPLATES = [
    ("{name}去哪里了？", "{name}去了{place}。"),
    ("{name}在做什么？", "{name}正在{act}。"),
    ("今天天气怎么样？", "今天天气{weather}。"),
    ("{name}感觉怎么样？", "{name}感觉很{feel}。"),
    ("{name}的朋友是谁？", "{name}的朋友是{name2}。"),
    ("介绍一下{animal}", "{animal}是一种可爱的小动物，{name}很喜欢它。"),
    ("什么时候去{place}？", "{time}去{place}最好。"),
    ("{name}喜欢做什么？", "{name}喜欢{act}。"),
]


def gen_story() -> str:
    tpl = random.choice(STORY_TEMPLATES)
    return tpl.format(
        name=random.choice(NAMES),
        name2=random.choice(NAMES),
        animal=random.choice(ANIMALS),
        place=random.choice(PLACES),
        act=random.choice(ACTIVITIES),
        weather=random.choice(WEATHERS),
        feel=random.choice(FEELINGS),
        time=random.choice(TIMES),
    )


def gen_qa() -> dict:
    q_tpl, a_tpl = random.choice(QA_TEMPLATES)
    ctx = dict(
        name=random.choice(NAMES),
        name2=random.choice(NAMES),
        animal=random.choice(ANIMALS),
        place=random.choice(PLACES),
        act=random.choice(ACTIVITIES),
        weather=random.choice(WEATHERS),
        feel=random.choice(FEELINGS),
        time=random.choice(TIMES),
    )
    return {
        "messages": [
            {"role": "user", "content": q_tpl.format(**ctx)},
            {"role": "assistant", "content": a_tpl.format(**ctx)},
        ]
    }


# 预训练语料: 10000 条短句
pretrain_path = OUT_DIR / "pretrain_sample.txt"
with open(pretrain_path, "w", encoding="utf-8") as f:
    for _ in range(10000):
        f.write(gen_story() + "\n")
print(f"已生成 {pretrain_path}  ({pretrain_path.stat().st_size/1024:.1f} KB)")

# SFT 数据: 2000 条对话
sft_path = OUT_DIR / "sft_sample.jsonl"
with open(sft_path, "w", encoding="utf-8") as f:
    for _ in range(2000):
        f.write(json.dumps(gen_qa(), ensure_ascii=False) + "\n")
print(f"已生成 {sft_path}  ({sft_path.stat().st_size/1024:.1f} KB)")
