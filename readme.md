# 心理健康智能问答系统

心理健康智能问答系统是一个基于深度学习的问答系统，专门针对心理健康领域设计。本项目通过数据预处理、模型训练和模型蒸馏等步骤构建了一个高效的问答模型。基于本地Qwen2.5-0.5B-Instruct模型进行微调。

## 目录结构

```
Mental_Health_Intelligent_QA2/
├── dataset/              # 数据集目录
├── output/               # 模型输出目录
├── logs/                 # 训练日志目录
├── src/                  # 源代码目录
│   ├── preprocess.py     # 数据预处理脚本
│   ├── train.py          # 模型训练脚本
│   ├── qa_dataset.py     # 数据集处理脚本
│   ├── token_analysis.py # Token分析脚本
│   ├── original_test.py  # 原始模型测试脚本
│   └── distilled_test.py # 蒸馏模型测试脚本
└── README.md             # 项目说明文档
```

## 功能特点

- **数据预处理**: 清洗并格式化心理领域问答数据
- **模型训练**: 基于指定数据集对模型进行微调
- **训练监控**: 支持使用TensorBoard查看训练损失
- **模型测试**: 提供原生模型与蒸馏后模型的本地服务接口供效果对比
- **Token分析**: 可视化输入序列长度分布，辅助超参设置

## 环境准备

在开始之前，请确保创建必要的目录：

```bash
mkdir -p dataset output logs
```

## 使用流程

### 1. 数据预处理

首次运行项目时，需要对数据进行预处理：

```bash
python src/preprocess.py
```

> 注意：此步骤只需执行一次。

### 2. Token分布分析（可选）

查看数据集中的token分布情况：

```bash
python src/token_analysis.py
```

### 3. 模型训练

执行主要的模型训练过程：

```bash
python src/train.py
```

训练过程中可以通过TensorBoard监控loss变化：

```bash
tensorboard --logdir=logs --bind_all
```

### 4. 原始模型测试（可选）

在浏览器中测试原始模型的效果：

```bash
python src/original_test.py
```

然后访问 `http://localhost:7860` 进行问答测试。

### 5. 蒸馏模型测试

关闭原始模型测试服务后，运行蒸馏后的模型进行对比：

```bash
python src/distilled_test.py
```

通过相同的问题测试，可以明显感受到模型优化后的效果提升。

## 技术细节

### 数据处理
- 输入数据来自 `psychology-10k-Deepseek-R1-zh/distill_psychology-10k-r1.json`
- 数据分为90%训练集和10%验证集，分别存储在 `dataset/train.json` 和 `dataset/val.json` 中
- 数据格式包含[input]和[output]字段，其中output由reasoning_content和content组成

### 模型训练
- 基于本地Qwen2.5-0.5B-Instruct模型进行微调
- 使用最大长度为2048的序列进行训练
- 默认训练5个epoch，批大小为1，梯度累积步数为64
- 使用AdamW优化器，初始学习率为1e-4
- 使用ReduceLROnPlateau调度器，在验证损失 plateau 时降低学习率

### 模型蒸馏
- 通过对比原始模型和蒸馏模型的效果展示知识蒸馏的有效性
- 蒸馏后的模型保存在 `output/best` 和 `output/last` 目录中

## 注意事项

- 请按照上述步骤顺序执行项目
- 确保各目录具有适当的读写权限
- 训练过程可能需要较长时间，请耐心等待
- 当前实现强制使用CPU进行训练和推理