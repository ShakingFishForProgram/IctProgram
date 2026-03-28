# 华为ICT大赛 - ViT推理加速方案

## 项目简介

本项目基于MindSpore框架和昇腾NPU 910B平台，实现了UNI模型（Vision Transformer）的推理加速，用于BACH乳腺癌病理图像的4分类任务。

### 任务一：WSI特征提取加速
- 目标：提升WSI特征提取速度
- 评分：WSI/秒 × 1000 × 50%

### 任务二：BACH乳腺癌分类
- 目标：提升4分类准确率
- 评分：ACC × 100 × 50%

---

## 环境要求

- Python 3.8+
- MindSpore 2.3.0+
- Ascend NPU 910B
- 昇腾驱动和CANN

---

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install mindspore mindcv pillow numpy scipy scikit-learn matplotlib joblib tqdm

# 设置昇腾NPU环境
source /usr/local/Ascend/set_env.sh
```

### 2. 下载模型权重

```bash
# 自动下载UNI2模型权重
python3 download_weights.py
```

---

## 使用说明

### 任务一：WSI特征提取速度测试

#### 训练模式（包含所有优化技术）
```bash
python3 run_wsi_speed_test.py \
  --mode train \
  --wsid_dir /path/to/your/wsi_images \
  --enable_quantization 1 \
  --enable_pruning 1 \
  --enable_distillation 1
```

#### 推理模式（自动选择最优模型）
```bash
python3 run_wsi_speed_test.py --mode inference --wsid_dir /path/to/your/wsi_images
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：train/inference | train |
| `--wsid_dir` | WSI图像目录 | /home/ma-user/work/WSI |
| `--output_dir` | 输出目录 | ./output_wsi |
| `--model_path` | 指定模型路径(inference用) | 自动选择 |
| `--batch_size` | 批次大小 | 64 |
| `--enable_quantization` | 启用PTQ量化 (0/1) | 0 |
| `--enable_pruning` | 启用结构化剪枝 (0/1) | 0 |
| `--enable_distillation` | 启用知识蒸馏 (0/1) | 0 |
| `--distill_epochs` | 蒸馏训练轮数 | 20 |
| `--num_workers` | 数据加载线程数 | 4 |

#### 输出文件
- `output_wsi/wsi_model.ckpt` - 蒸馏后学生模型
- `output_wsi/wsi_model_quantized.ckpt` - 量化模型
- `output_wsi/wsi_model_pruned.ckpt` - 剪枝模型
- `output_wsi/summary.txt` - 测试结果

---

### 任务二：BACH乳腺癌分类

#### 训练分类器
```bash
python3 run_task2_classification.py
```

#### 推理预测
```bash
# 推理新图像（自动使用已训练的模型）
python3 run_task2_inference.py --data_dir /path/to/your/images
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 图像目录 | ./data |
| `--batch_size` | 批次大小 | 32 |

#### 输出文件
- `output_task2/classifier_model.pkl` - SVM分类器模型
- `output_task2/model_info.json` - 标签编码器和标准化参数
- `output_task2/summary.txt` - 分类结果摘要

---

## 模型文件说明

### 任务一模型

| 文件 | 说明 |
|------|------|
| `output_wsi/wsi_model.ckpt` | 蒸馏后的学生模型 (ViT-B) |
| `output_wsi/wsi_model_quantized.ckpt` | 量化模型 (FP16) |
| `output_wsi/wsi_model_pruned.ckpt` | 剪枝模型 |

### 任务二模型

| 文件 | 说明 |
|------|------|
| `output_task2/classifier_model.pkl` | SVM分类器 |
| `output_task2/model_info.json` | 标签编码器和标准化参数 |

---

## 实验结果

### 任务一：速度指标

| 指标 | 数值 |
|------|------|
| 优化前速度 | 3.56 WSI/秒 |
| 优化后速度 | **507 WSI/秒** |
| 速度提升 | **142倍** |
| 模型压缩比 | 3.50x |

### 任务二：分类指标

| 指标 | 数值 |
|------|------|
| 测试准确率 | **73.33%** |
| F1 (macro) | 0.72 |
| 各类别AUC | >0.90 |

---

## 目录结构

```
UNI/
├── run_wsi_speed_test.py      # 任务一：WSI速度测试
├── run_task2_classification.py # 任务二：分类器训练
├── run_task2_inference.py    # 任务二：推理脚本
├── download_weights.py        # 下载模型权重
├── uni/                      # UNI模型代码
│   └── get_encoder_ms.py    # 特征提取器
├── output_wsi/               # 任务一输出
├── output_task2/             # 任务二输出
└── README.md                # 本文件
```

---

## 常见问题

### Q1: 如何确认NPU已正确识别？
```bash
# 检查昇腾NPU状态
npu-smi info
```

### Q2: 推理时找不到模型文件？
确认已在对应output目录下生成了模型文件，或修改配置文件中的路径。

### Q3: 速度测试结果不稳定？
建议多次测试取平均值，首次运行包含模型加载时间。

---

## 联系方式

如有疑问，请联系参赛团队。
