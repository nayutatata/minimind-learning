# MiniMind 项目开发指南

## 项目概述

MiniMind 是一个轻量级大语言模型训练框架，从头开始仅用 3 块钱 + 2 小时即可训练出 25.8M 参数的超小语言模型。项目包含完整的预训练、监督微调、LoRA、DPO、GRPO/PPO 等训练方案，所有核心代码均基于 PyTorch 原生实现。

## 环境设置

### 前置要求
- Python 3.8+
- CUDA 11.8+ （推荐用于 GPU 加速）
- 依赖管理：使用 `conda` 或 `pip` 安装 `requirements.txt`

### 快速开始

1. **创建虚拟环境**（推荐使用 conda）
   ```bash
   conda create -n minimind python=3.10
   conda activate minimind
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **验证安装**
   - 检查 PyTorch 和 CUDA：`python -c "import torch; print(torch.cuda.is_available())"`
   - 检查关键包：`pip list | grep -E "transformers|peft|trl"`

### 关键依赖说明
- **transformers==4.57.1**: Hugging Face 核心库，提供预训练模型和 tokenizer
- **peft==0.7.1**: PEFT（Prompt Engineering for Training）用于 LoRA 微调
- **trl==0.13.0**: Transformers Reinforcement Learning，支持 DPO、PPO、GRPO
- **torch==2.6.0**: PyTorch 核心框架
- **wandb**: 实验追踪和可视化
- **streamlit**: Web UI 演示

### 环境验证清单
- [ ] Python 版本 >= 3.8
- [ ] PyTorch 安装且能访问 GPU/CPU
- [ ] `transformers` 库可正常导入
- [ ] 项目根目录的 `model/`、`dataset/`、`trainer/` 文件夹完整

---

## 代码结构理解

### 目录架构

```
minimind-learning/
├── model/                  # 模型定义
│   ├── model_minimind.py   # 核心模型架构
│   ├── model_lora.py       # LoRA 微调适配
│   ├── tokenizer.json      # 词表配置
│   └── tokenizer_config.json
├── trainer/                # 训练脚本
│   ├── train_pretrain.py   # 预训练
│   ├── train_full_sft.py   # 全参数 SFT
│   ├── train_lora.py       # LoRA 微调
│   ├── train_dpo.py        # DPO 对齐
│   ├── train_ppo.py        # PPO 强化学习
│   ├── train_grpo.py       # GRPO 强化学习
│   ├── train_reason.py     # 推理能力训练
│   ├── train_distillation.py # 知识蒸馏
│   └── trainer_utils.py    # 共享工具函数
├── dataset/                # 数据处理
│   ├── lm_dataset.py       # 数据集类定义
│   └── dataset.md          # 数据集文档
├── scripts/                # 推理和部署脚本
│   ├── chat_openai_api.py  # OpenAI API 适配层
│   ├── web_demo.py         # Streamlit 演示
│   └── convert_model.py    # 模型转换工具
├── requirements.txt        # 依赖清单
└── eval_llm.py            # 评估脚本
```

### 核心模块解读

#### 1. **model_minimind.py**
- 定义 MiniMind 核心架构：Transformer 层
- 关键类：`MiniMind` 模型主体
- 重点关注：`forward()` 方法、attention 机制、MLP 层设计
- **学习建议**：从 `__init__()` 开始，理解参数维度和层间连接

#### 2. **model_lora.py**
- 基于 PEFT 库实现 LoRA 低秩自适应
- 当内存不足或需快速微调时使用
- **关键参数**：
  - `lora_rank`: 低秩矩阵维度（通常 8-64）
  - `lora_alpha`: 缩放因子
  - `target_modules`: 目标层名称（如 `"q_proj"`、`"v_proj"`）

#### 3. **trainer_utils.py**
- 共享的训练工具：优化器设置、学习率调度、loss 计算
- 包含数据加载器、分布式训练配置等通用函数
- **常用函数**：`setup_optimizer()`、`get_lr_scheduler()`、`setup_distributed()`

#### 4. **dataset/lm_dataset.py**
- 自定义 PyTorch Dataset 类
- 处理 tokenization、序列打包、数据增强
- **重点**：理解 batch 构造逻辑和 token ID 转换

#### 5. **其他训练脚本**
- **train_pretrain.py**: 从零开始的自监督学习（Causal LM）
- **train_full_sft.py**: 指令跟随微调（全参数更新）
- **train_dpo.py**: 直接偏好优化（对齐 AI 输出与人类偏好）
- **train_ppo.py/train_grpo.py**: 强化学习对齐（奖励模型基础）

---

## 代码审查与最佳实践

### 1. 训练脚本规范

#### ✓ 推荐做法
- **参数化配置**：使用 `argparse` 或配置文件管理超参数，避免硬编码
  ```python
  parser.add_argument('--learning_rate', type=float, default=5e-4)
  parser.add_argument('--batch_size', type=int, default=32)
  ```
- **Checkpointing**：定期保存模型和优化器状态
  ```python
  torch.save({
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'step': step
  }, checkpoint_path)
  ```
- **日志记录**：使用 `wandb` 或 `tensorboard` 跟踪训练指标
  ```python
  wandb.log({'loss': loss.item(), 'learning_rate': lr})
  ```

#### ✗ 避免的做法
- 硬编码路径或超参数（使用 CLI 参数或配置文件）
- 训练循环中的 `print()` 而不是日志库
- 不检查 GPU 内存，导致 OOM 崩溃
- 忘记设置随机种子（影响复现性）

### 2. 模型修改指南

#### 编辑模型时的检查清单
- [ ] 修改参数维度后，验证张量形状兼容性
- [ ] 添加新层后，确保梯度能正确反向传播（`loss.backward()`）
- [ ] 改变输入/输出接口时，同步更新所有调用处
- [ ] 使用 `model.train()` 和 `model.eval()` 正确切换模式（影响 Dropout、BatchNorm）

#### 常见错误示例
```python
# ✗ 错误：维度不匹配
output = self.linear(hidden)  # hidden: (B, seq, 768), linear 期望 (B, 768)

# ✓ 正确：确保维度一致
output = self.linear(hidden.view(-1, hidden.size(-1))).view(B, -1, hidden.size(-1))

# ✗ 错误：忘记设置设备
model = MiniMind()  # 在 CPU 上
x = x.cuda()
y = model(x)  # RuntimeError: 设备不匹配

# ✓ 正确：显式管理设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniMind().to(device)
x = x.to(device)
```

### 3. 数据处理规范

#### 数据集设计原则
- **Tokenization 一致性**：确保训练和推理使用同一 tokenizer
- **序列长度管理**：统一序列长度或使用动态 padding
- **数据验证**：检查数据类型、范围和缺失值
  ```python
  assert all(token_id < vocab_size for token_id in batch['input_ids'])
  ```

#### 推荐实现
```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = load_json(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'][0]
```

### 4. 性能优化建议

| 问题 | 解决方案 |
|------|--------|
| 显存占用过高 | 减小 `batch_size`，使用 LoRA 替代全参数微调，启用 `gradient_checkpointing` |
| 训练速度慢 | 使用混合精度训练 (`torch.cuda.amp.autocast()`)，多 GPU 分布式训练 |
| 模型过拟合 | 增加数据，使用 Dropout，添加 L2 正则化 |
| Loss 不下降 | 调整学习率，检查数据质量，验证梯度流动 |

### 5. 提交代码前的清单

- [ ] 代码能正常运行（测试至少一个小 batch）
- [ ] 所有 import 都被使用（清理未使用的导入）
- [ ] 遵循 PEP 8 风格（变量名、函数名小写加下划线）
- [ ] 添加必要的注释和文档字符串
- [ ] 处理异常情况（文件不存在、维度错误等）
- [ ] 提交前运行 linter（如 `flake8`、`black`）

---

## 常见问题排查

### Q: GPU 显存不足（CUDA out of memory）
**A**: 
1. 减小 `batch_size`（如从 32 到 8）
2. 使用 LoRA 微调代替全参数更新
3. 启用梯度累积或混合精度训练
4. 检查是否有多个模型实例占用显存

### Q: 模型输出为 NaN
**A**:
1. 检查学习率是否过高
2. 验证输入数据是否正确规范化
3. 查看是否有梯度爆炸（打印 grad norm）
4. 尝试使用梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

### Q: 训练过程中 loss 不下降
**A**:
1. 确认数据加载器工作正常（打印几个 batch）
2. 验证模型是否处于 `train()` 模式
3. 检查学习率scheduler 是否有问题
4. 尝试从零初始化模型参数

---

## 学习资源

- Hugging Face 官方教程：https://huggingface.co/docs/transformers/
- PEFT LoRA 说明：https://huggingface.co/docs/peft/
- PyTorch 分布式训练：https://pytorch.org/docs/stable/distributed.html

---

## 开发工作流

1. **阅读文档**：先看 [README.md](../README.md) 和模块内注释
2. **理解现有代码**：追踪一个训练脚本的执行流程
3. **逐步修改**：每次改一个小模块，用小数据测试
4. **验证改动**：对比前后的性能指标（loss、精度等）
5. **提交更改**：确保代码能在本地完整运行

---

**记住**：这是一个教学项目，不追求工程上的完美，重点是理解大模型的每一步原理！遇到问题时，查阅代码注释和相关论文，深入理解背后的算法逻辑。
