# MiniMind Jupyter Notebooks

本项目将 [MiniMind](https://github.com/jingyaogong/minimind) 大语言模型训练流程复刻为 Jupyter Notebook 格式，支持交互式训练与推理。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| [`minimind_training.ipynb`](minimind_training.ipynb) | 本地训练 Notebook，适用于本地 GPU 环境 |
| [`minimind_colab.ipynb`](minimind_colab.ipynb) | Google Colab 专用 Notebook，支持一键运行 |

## 🚀 快速开始

### 方式一：Google Colab（推荐）

1. 上传 `minimind_colab.ipynb` 到 Google Colab
2. 选择 GPU 运行时（Runtime > Change runtime type > GPU）
3. 按顺序运行所有单元格

### 方式二：本地运行

```bash
# 1. 安装依赖
pip install torch transformers datasets matplotlib

# 2. 准备数据
# 从 https://huggingface.co/datasets/jingyaogong/minimind_dataset 下载数据
# 放入 ./minimind-master/dataset/ 目录

# 3. 启动 Jupyter
jupyter notebook minimind_training.ipynb
```

## 📚 Notebook 内容结构

两个 Notebook 都包含以下完整训练流程：

| 部分 | 内容 |
|------|------|
| 第一部分 | 环境准备与依赖检查 |
| 第二部分 | 模型架构定义（MiniMindConfig, Attention, FeedForward, MoE 等） |
| 第三部分 | 工具函数（种子设置、学习率调度、日志等） |
| 第四部分 | 数据集类（PretrainDataset, SFTDataset） |
| 第五部分 | 预训练 (Pretrain) |
| 第六部分 | 指令微调 (SFT) |
| 第七部分 | GRPO 强化学习训练 |
| 第八部分 | PPO 强化学习训练 |
| 第九部分 | DPO 偏好对齐训练 |
| 第十部分 | LoRA 低秩微调 |
| 第十一部分 | 模型蒸馏 |
| 第十二部分 | 模型推理与对话 |

## 📊 包含的对齐方法

| 方法 | 类型 | 说明 |
|------|------|------|
| DPO | 偏好对齐 | 直接偏好优化，使用 chosen/rejected 对 |
| GRPO | 强化学习 | 组相对策略优化，无需 Critic 模型 |
| PPO | 强化学习 | 经典 Actor-Critic 方法 |
| LoRA | 参数微调 | 低秩适配，减少可训练参数 |
| 蒸馏 | 知识迁移 | KL 散度模仿教师模型输出 |

## 📥 数据准备

### 训练数据

从 [HuggingFace - minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 下载：

| 文件 | 大小 | 用途 |
|------|------|------|
| `pretrain_t2t_mini.jsonl` | ~1.2GB | 预训练数据 |
| `sft_t2t_mini.jsonl` | ~200MB | 指令微调数据 |
| `dpo.jsonl` | ~53MB | DPO 偏好数据 |
| `rlaif.jsonl` | ~86MB | RL 训练数据 |

### 数据格式

**预训练数据** (`pretrain_t2t_mini.jsonl`):
```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易..."}
{"text": "清晨的阳光透过窗帘洒进房间..."}
```

**SFT 数据** (`sft_t2t_mini.jsonl`):
```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"}
    ]
}
```

**DPO 数据** (`dpo.jsonl`):
```json
{
    "chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good answer"}],
    "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad answer"}]
}
```

## 💻 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 8GB VRAM | 16GB+ VRAM (如 RTX 3090/4090) |
| 内存 | 16GB | 32GB+ |
| 存储 | 10GB | 20GB+ |

### Colab 配置

- **免费 T4 GPU**: 16GB VRAM，可完成全部训练
- **预计训练时间**: 1-2 小时（预训练 + SFT）

## 🔧 训练流程

```
预训练 → SFT → DPO → GRPO → PPO → LoRA/蒸馏 → 推理
```

### 最小训练流程

如果只想快速复现对话模型：

1. 运行预训练单元格
2. 运行 SFT 单元格
3. 运行推理单元格

### 完整训练流程

按顺序运行所有单元格，包含全部对齐方法。

## 📝 模型配置

默认模型配置（MiniMind-3）：

| 参数 | 值 |
|------|------|
| hidden_size | 768 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 4 |
| vocab_size | 6400 |
| 参数量 | ~64M |

## 📄 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。

## 🙏 致谢

- 原始项目: [minimind](https://github.com/jingyaogong/minimind) by Jingyao Gong
- 数据集: [minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)

## 📖 引用

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
}
```
