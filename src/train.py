# train.py
# 微调训练脚本

# -*- coding: utf-8 -*-
import os.path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from qa_dataset import QADataset
from tqdm import tqdm
import time, sys
import json # 导入 json 模块用于读取数据集文件

def train_model(model, train_loader, val_loader, optimizer,
                device, num_epochs, model_output_dir, scheduler, writer, gradient_accumulation_steps):
    batch_step, best_loss = 0, 100
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            # 反向传播，累计当前梯度
            loss.backward()
            # 每gradient_accumulation_steps步，或者到达epoch最后一个batch时，更新参数
            if (index + 1) % gradient_accumulation_steps == 0 or index == len(train_loader) - 1:
                # 梯度裁剪（可选，防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 更新网络参数
                optimizer.step()
                # 清空梯度，为下一轮累积做准备
                optimizer.zero_grad()
            writer.add_scalar('Loss/train', loss, batch_step)
            batch_step += 1
        time2 = time.time()
        tqdm.write(f"{index}, epoch: {epoch} -loss: {str(loss)} ; each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")
        # 学习率调整
        scheduler.step(val_loss)
        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(model_output_dir, "best")
            print("Save Best Model To ", best_model_path, ", epoch: ", epoch)
            model.save_pretrained(best_model_path)
        # 保存当前模型
        last_model_path = os.path.join(model_output_dir, "last")
        print("Save Last Model To ", last_model_path, ", epoch: ", epoch)
        model.save_pretrained(last_model_path)

def validate_model(model, device, val_loader):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)

def main():
    # 基础模型位置
    model_name = "D:/project/qwen2.5-0.5B-Instruct"
    # 训练集
    train_json_path = "./dataset/train.json"
    # 验证集
    val_json_path = "./dataset/val.json"
    # 最大输出长度
    max_length = 2048
    # 训练周期
    epochs = 5
    batch_size = 1
    # 梯度累计步数
    gradient_accumulation_steps = 64
    lr = 1e-4
    model_output_dir = "./output"
    logs_dir = "./logs"
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 强制设置为 CPU，因为用户明确表示只有 CPU
    if str(device).startswith("cuda"):
        device = torch.device("cpu")
        print("CUDA available but forced to use CPU.")
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    # ----------------------------------------------------
    # 核心修改：加载数据并进行切片以加速训练
    # ----------------------------------------------------
    print("Start Load Train Data...")
    try:
        # 读取JSONL格式文件
        train_data = []
        with open(train_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    try:
                        train_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # 忽略解析错误的行
        # 仅取前100个训练样本
        train_data_subset = train_data[:100]
        print(f"Original training size: {len(train_data)}, using subset size: {len(train_data_subset)}")
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_json_path}")
        return

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 4,
    }
    # 使用切片后的数据子集初始化 QADataset
    training_set = QADataset(train_data_subset, tokenizer, max_length)
    training_loader = DataLoader(training_set, **train_params)

    print("Start Load Validation Data...")
    try:
        # 读取JSONL格式文件
        val_data = []
        with open(val_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    try:
                        val_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # 忽略解析错误的行
        # 仅取前50个验证样本
        val_data_subset = val_data[:50]
        print(f"Original validation size: {len(val_data)}, using subset size: {len(val_data_subset)}")
    except FileNotFoundError:
        print(f"Error: Validation file not found at {val_json_path}")
        return

    val_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 4,
    }
    # 使用切片后的数据子集初始化 QADataset
    val_set = QADataset(val_data_subset, tokenizer, max_length)
    val_loader = DataLoader(val_set, **val_params)
    # ----------------------------------------------------
    # 核心修改结束
    # ----------------------------------------------------

    # 日志记录
    writer = SummaryWriter(logs_dir)
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    # 学习率调度器，连续两个周期没有改进，学习率调整为当前学习率的80%
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.8)
    model = model.to(device)
    # 开始训练
    print("Start Training...")
    train_model(model=model,
                train_loader=training_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                num_epochs=epochs,
                model_output_dir=model_output_dir,
                scheduler=scheduler,
                writer=writer,
                gradient_accumulation_steps=gradient_accumulation_steps)
    writer.close()

if __name__ == '__main__':
    main()