#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore版CLAM分类器训练脚本（任务二）
=====================================
功能：使用提取的特征训练CLAM分类器，评估分类精度
评估指标：ACC (准确率), F1-Score

使用方法：
    python train_classifier_ms.py --features_path ./output_ms/features.npy
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore.train import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='MindSpore CLAM分类器训练')
    
    parser.add_argument('--features_path', type=str, default=None,
                       help='特征文件路径 (.npy)')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='标签文件路径 (.npy)')
    
    parser.add_argument('--classifier', type=str, default='clam', 
                       choices=['clam', 'lr'],
                       help='分类器类型')
    parser.add_argument('--n_classes', type=int, default=4,
                       help='类别数量')
    
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    parser.add_argument('--clam_epochs', type=int, default=50,
                       help='CLAM训练轮数')
    parser.add_argument('--clam_lr', type=float, default=1e-3,
                       help='CLAM学习率')
    parser.add_argument('--clam_hidden', type=int, default=128,
                       help='CLAM隐藏层维度')
    
    parser.add_argument('--output_dir', type=str, default='./output_ms',
                       help='输出目录')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='保存模型')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='保存可视化图表')
    
    return parser.parse_args()


class AttentionPool(nn.Cell):
    """注意力池化层"""
    
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.SequentialCell([
            nn.Dense(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dense(hidden_dim, 1)
        ])
    
    def construct(self, x):
        attn_weights = self.attention(x)
        attn_weights = ops.Softmax(axis=0)(attn_weights)
        weighted = x * attn_weights
        return weighted.sum(axis=0)


class CLAMClassifier(nn.Cell):
    """CLAM分类器 (MindSpore版本)"""
    
    def __init__(self, in_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.n_classes = n_classes
        self.in_dim = in_dim
        
        self.attention = AttentionPool(in_dim, hidden_dim)
        
        self.instance_classifiers = nn.CellList([
            nn.SequentialCell([
                nn.Dense(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Dense(hidden_dim, 2)
            ]) for _ in range(n_classes)
        ])
    
    def construct(self, x):
        if x.ndim == 2:
            x = x.expand_dims(0)
        
        bag_logits = []
        
        for c in range(self.n_classes):
            inst_logits = self.instance_classifiers[c](x)
            bag_pred = inst_logits[:, 1]
            bag_logits.append(bag_pred)
        
        bag_logits = ops.Stack(axis=1)(bag_logits)
        return bag_logits
    
    def get_attention(self, x):
        if x.ndim == 2:
            x = x.expand_dims(0)
        return self.attention(x)


class CLAMDataset():
    """CLAM数据集"""
    
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_dataset(features, labels, batch_size=32, shuffle=True):
    """创建MindSpore数据集"""
    dataset = ms.dataset.NumpySlicesDataset(
        data=(features, labels),
        column_names=["features", "labels"],
        shuffle=shuffle
    )
    dataset = dataset.batch(batch_size, per_batch_size=batch_size)
    return dataset


def load_features_from_npy(features_path, labels_path):
    """从npy文件加载特征"""
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  MindSpore CLAM分类器训练 - 任务二（精度评估）")
    print("=" * 60)
    print(f"分类器: {args.classifier}")
    print(f"训练比例: {args.train_ratio}")
    print(f"验证比例: {args.val_ratio}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[1/4] 加载数据...")
    
    if args.features_path and args.labels_path:
        features, labels = load_features_from_npy(args.features_path, args.labels_path)
    else:
        features_path = os.path.join(args.output_dir, 'features.npy')
        labels_path = os.path.join(args.output_dir, 'labels.npy')
        
        if os.path.exists(features_path) and os.path.exists(labels_path):
            features, labels = load_features_from_npy(features_path, labels_path)
        else:
            print("错误: 请先运行特征提取脚本")
            sys.exit(1)
    
    print(f"特征形状: {features.shape}")
    print(f"类别数量: {len(np.unique(labels))}")
    
    print("\n[2/4] 划分数据集...")
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=test_ratio,
        random_state=args.random_seed,
        stratify=labels
    )
    
    val_size = args.val_ratio / (args.train_ratio + args.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=args.random_seed,
        stratify=y_temp
    )
    
    print(f"训练集: {len(X_train)}")
    print(f"验证集: {len(X_val)}")
    print(f"测试集: {len(X_test)}")
    
    print("\n[3/4] 训练分类器...")
    
    class_names = ['Normal', 'Benign', 'In situ', 'Invasive']
    
    if args.classifier == 'clam':
        ms.set_context(device_target='Ascend', mode=ms.GRAPH_MODE)
        
        train_dataset = create_dataset(X_train, y_train, batch_size=32, shuffle=True)
        val_dataset = create_dataset(X_val, y_val, batch_size=32, shuffle=False)
        test_dataset = create_dataset(X_test, y_test, batch_size=32, shuffle=False)
        
        model = CLAMClassifier(
            in_dim=features.shape[1],
            n_classes=args.n_classes,
            hidden_dim=args.clam_hidden
        )
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = nn.Adam(model.trainable_params(), learning_rate=args.clam_lr)
        
        train_network = nn.TrainOneStepCell(loss_fn, optimizer)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        best_model_params = None
        
        for epoch in range(args.clam_epochs):
            model.set_train(True)
            total_loss = 0
            
            for batch in train_dataset:
                features_batch = batch[0]
                labels_batch = batch[1]
                
                loss = train_network(features_batch, labels_batch)
                total_loss += loss.asnumpy()
            
            avg_loss = total_loss / len(train_dataset)
            
            def evaluate(model, dataset):
                model.set_train(False)
                all_preds = []
                all_labels = []
                
                for batch in dataset:
                    features_batch = batch[0]
                    labels_batch = batch[1]
                    
                    logits = model(features_batch)
                    preds = logits.argmax(axis=1)
                    
                    all_preds.extend(preds.asnumpy())
                    all_labels.extend(labels_batch.asnumpy())
                
                return accuracy_score(all_labels, all_preds)
            
            train_acc = evaluate(model, train_dataset)
            val_acc = evaluate(model, val_dataset)
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_params = model.parameters_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.clam_epochs} - Loss: {avg_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if best_model_params:
            model.load_parameters(best_model_params)
        
        model.set_train(False)
        all_preds = []
        all_labels = []
        
        for batch in test_dataset:
            features_batch = batch[0]
            labels_batch = batch[1]
            logits = model(features_batch)
            preds = logits.argmax(axis=1)
            all_preds.extend(preds.asnumpy())
            all_labels.extend(labels_batch.asnumpy())
        
        test_preds = np.array(all_preds)
        test_labels = np.array(all_labels)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        
    elif args.classifier == 'lr':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        test_preds = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        
        history = None
        
    else:
        raise ValueError(f"不支持的分类器: {args.classifier}")
    
    print("\n[4/4] 评估结果...")
    
    print("\n" + "=" * 60)
    print("  测试集评估结果")
    print("=" * 60)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1-Score (macro): {test_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, test_preds, target_names=class_names))
    
    if args.save_plots:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_test, test_preds, class_names, cm_path)
        print(f"\n混淆矩阵已保存: {cm_path}")
        
        if history:
            history_path = os.path.join(args.output_dir, 'training_history.png')
            plot_training_history(history, history_path)
            print(f"训练历史已保存: {history_path}")
    
    results = {
        'classifier': args.classifier,
        'accuracy': float(test_acc),
        'f1_macro': float(test_f1),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
    }
    
    results_path = os.path.join(args.output_dir, 'classification_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_path}")
    
    print("\n" + "=" * 60)
    print("  最终得分计算")
    print("=" * 60)
    precision_score = test_acc * 100
    print(f"精度分 (ACC × 100): {precision_score:.2f}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
