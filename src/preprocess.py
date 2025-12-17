# preprocess.py
# 数据预处理、拆分脚本

import json
from tqdm import tqdm

data_path = "./psychology-10k-Deepseek-R1-zh/distill_psychology-10k-r1.json"
datas = []
with open(data_path, "r", encoding="utf-8") as r:
    for line in r:
        line = json.loads(line)
        input = line["input"]
        content = line["content"]
        reasoning_content = line["reasoning_content"]
        ## 将思考和答案整合成output字段
        output = f"{reasoning_content}{content}"
        datas.append({"input": input, "output": output})

## 90%数据训练，10%数据验证
with open("./dataset/train.json", "w", encoding="utf-8") as train_w:
    for item in tqdm(datas[:int(len(datas)*0.9)]):
        line = json.dumps(item, ensure_ascii=False)
        train_w.write(line + "\n")
        train_w.flush()

with open("./dataset/val.json", "w", encoding="utf-8") as val_w:
    for item in tqdm(datas[int(len(datas)*0.9):]):
        line = json.dumps(item, ensure_ascii=False)
        val_w.write(line + "\n")
        val_w.flush()