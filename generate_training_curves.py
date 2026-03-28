#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成任务二训练曲线图
用于PPT展示
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 数据（从train_log.txt读取）
# 这是多模型版本的训练结果
configs = [
    'SVM-RBF-C1', 'SVM-RBF-C10', 'SVM-RBF-C0.1', 'SVM-Poly-C1', 'SVM-Linear',
    'SVM-RBF-balanced', 'SVM-Linear-balanced',
    'RF-n100-d10', 'RF-n200-d15', 'RF-n100-dNone',
    'GB-n100-lr0.1', 'GB-n150-lr0.05',
    'Voting-Ensemble'
]

train_acc = [
    0.9821, 1.0000, 0.6308, 0.9928, 1.0000,
    0.9821, 1.0000,
    1.0000, 1.0000, 1.0000,
    1.0000, 1.0000,
    1.0000
]

val_acc = [
    0.7167, 0.7167, 0.5333, 0.6833, 0.7333,
    0.7167, 0.7333,
    0.6333, 0.6833, 0.6000,
    0.6667, 0.6167,
    0.7333
]

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 训练和验证准确率对比
ax1 = axes[0, 0]
x = np.arange(len(configs))
width = 0.35
bars1 = ax1.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#1f77b4', alpha=0.8)
bars2 = ax1.bar(x + width/2, val_acc, width, label='Val Accuracy', color='#ff7f0e', alpha=0.8)
ax1.set_xlabel('Model Configuration', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Comparison: Train vs Validation Accuracy', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
ax1.legend(loc='lower right')
ax1.set_ylim([0.4, 1.1])
ax1.axhline(y=max(val_acc), color='green', linestyle='--', alpha=0.7, label=f'Best: {max(val_acc):.4f}')

# 图2: 验证准确率趋势
ax2 = axes[0, 1]
ax2.plot(x, val_acc, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Val Acc')
ax2.fill_between(x, val_acc, alpha=0.3, color='#2ca02c')
ax2.set_xlabel('Model Configuration Index', fontsize=12)
ax2.set_ylabel('Validation Accuracy', fontsize=12)
ax2.set_title('Validation Accuracy Trend', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
ax2.axhline(y=max(val_acc), color='red', linestyle='--', alpha=0.7)
ax2.annotate(f'Best: {max(val_acc):.4f}', 
             xy=(val_acc.index(max(val_acc)), max(val_acc)), 
             xytext=(val_acc.index(max(val_acc))+1, max(val_acc)+0.05),
             fontsize=10, color='red')

# 图3: 各类别F1分数
ax3 = axes[1, 0]
classes = ['Benign', 'InSitu', 'Invasive', 'Normal']
f1_scores = [0.48, 0.79, 0.88, 0.73]
auc_scores = [0.74, 0.92, 0.92, 0.95]

x = np.arange(len(classes))
width = 0.35
bars3 = ax3.bar(x - width/2, f1_scores, width, label='F1 Score', color='#17becf', alpha=0.8)
bars4 = ax3.bar(x + width/2, auc_scores, width, label='AUC Score', color='#9467bd', alpha=0.8)
ax3.set_xlabel('Class', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Per-Class Performance: F1 and AUC', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(classes, fontsize=11)
ax3.legend(loc='upper right')
ax3.set_ylim([0, 1.1])

# 添加数值标签
for bar, score in zip(bars3, f1_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.2f}', ha='center', va='bottom', fontsize=10)
for bar, score in zip(bars4, auc_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.2f}', ha='center', va='bottom', fontsize=10)

# 图4: 准确率总结
ax4 = axes[1, 1]
metrics = ['Test Acc', 'F1 (macro)', 'F1 (weighted)', 'Best Val Acc']
values = [0.7333, 0.7191, 0.7191, 0.7333]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax4.barh(metrics, values, color=colors, alpha=0.8)
ax4.set_xlabel('Score', fontsize=12)
ax4.set_title('Final Model Performance Metrics', fontsize=14, fontweight='bold')
ax4.set_xlim([0, 1.0])

# 添加数值标签
for bar, val in zip(bars, values):
    ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('E:/1AAA_Competition/2025HUAWEI_ICT/2025HUAWEI_ICT_ViT_CV/UNI_finish(1)/UNI/output_task2/training_curves_ppt.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("图表已保存!")

# 单独创建一个更清晰的训练曲线图
fig2, ax = plt.subplots(figsize=(14, 6))

# 配置分组
groups = ['SVM\n(5 configs)', 'SVM-Balanced\n(2 configs)', 'RandomForest\n(3 configs)', 'GradientBoosting\n(2 configs)', 'Voting\n(1 config)']
group_train = [0.9810, 0.9910, 1.0, 1.0, 1.0]
group_val = [0.6767, 0.7250, 0.6389, 0.6417, 0.7333]

x = np.arange(len(groups))
width = 0.35

ax.bar(x - width/2, group_train, width, label='Train Acc', color='#3498db', alpha=0.8)
ax.bar(x + width/2, group_val, width, label='Val Acc', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Model Group', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Classification Model Groups - Training vs Validation Accuracy', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim([0.5, 1.1])

# 添加数值标签
for i, (train, val) in enumerate(zip(group_train, group_val)):
    ax.text(i - width/2, train + 0.02, f'{train:.2%}', ha='center', fontsize=10)
    ax.text(i + width/2, val + 0.02, f'{val:.2%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('E:/1AAA_Competition/2025HUAWEI_ICT/2025HUAWEI_ICT_ViT_CV/UNI_finish(1)/UNI/output_task2/model_group_comparison.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("分组对比图已保存!")