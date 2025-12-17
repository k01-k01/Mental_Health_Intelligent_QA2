# token_analysis.py
# 分析训练集中输出内容的token分布情况

import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 获取Token数
def get_num_tokens(file_path, tokenizer):
    input_num_tokens = []
    with open(file_path, "r", encoding="utf-8") as r:
        for line in r:
            line = json.loads(line)
            output = line["output"]
            tokens = len(tokenizer(output)["input_ids"])
            input_num_tokens.append(tokens)
    return input_num_tokens

# 计算分布
def count_intervals(num_tokens, interval):
    max_value = max(num_tokens)
    intervals_count = {}
    for lower_bound in range(0, max_value + 1, interval):
        upper_bound = lower_bound + interval
        count = len([num for num in num_tokens if lower_bound <= num < upper_bound])
        intervals_count[f"{lower_bound}-{upper_bound}"] = count
    return intervals_count

def main():
    model_path = "D:/project/qwen2.5-0.5B-Instruct"
    train_data_path = "./dataset/train.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_num_tokens = get_num_tokens(train_data_path, tokenizer)
    intervals_count = count_intervals(input_num_tokens, 128)
    print(intervals_count)
    x = [k for k, v in intervals_count.items()]
    y = [v for k, v in intervals_count.items()]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, y)
    plt.title('训练集Token分布/情况')
    plt.ylabel('数量')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')
    plt.show()

if __name__ == '__main__':
    main()