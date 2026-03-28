#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore完整运行脚本：任务一 + 任务二
=====================================
功能：一体化运行特征提取和CLAM分类器训练
评估：速度分 + 精度分

使用方法：
    python run_complete_ms.py --data_root ../Photos

评分计算：
    - 速度分 = 平均每秒提取WSI数量 × 10 × 50%
    - 精度分 = 平均ACC × 100 × 50%
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, ops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from uni.get_encoder_ms import get_encoder


def parse_args():
    parser = argparse.ArgumentParser(description='MindSpore UNI完整运行脚本')
    
    parser.add_argument('--model_name', type=str, default='uni2-h',
                       choices=['uni', 'uni2-h'],
                       help='模型名称')
    parser.add_argument('--data_root', type=str, default='../Photos',
                       help='数据根目录')
    parser.add_argument('--csv_file', type=str, default='microscopy_ground_truth.csv',
                       help='CSV文件')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='启用混合精度')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备')
    
    parser.add_argument('--classifier', type=str, default='clam',
                       choices=['clam', 'lr'],
                       help='分类器类型')
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
    
    parser.add_argument('--output_dir', type=str, default='./output_ms',
                       help='输出目录')
    
    return parser.parse_args()


class BACHImageDataset:
    """BACH图像数据集"""
    
    def __init__(self, data_root, csv_file, transforms=None):
        import pandas as pd
        
        self.data_root = Path(data_root)
        self.transforms = transforms
        
        csv_path = self.data_root / csv_file
        df = pd.read_csv(csv_path)
        df.columns = ['filename', 'label']
        
        self.image_paths = []
        self.labels = []
        
        for _, row in df.iterrows():
            img_path = self.data_root / row['label'] / row['filename']
            if img_path.exists():
                self.image_paths.append(str(img_path))
                self.labels.append(row['label'])
        
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            for trans in self.transforms:
                image = trans(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)
        
        return image, self.label_to_idx[label]


class AttentionPool(ms.nn.Cell):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attention = ms.nn.SequentialCell([
            ms.nn.Dense(in_dim, hidden_dim),
            ms.nn.Tanh(),
            ms.nn.Dense(hidden_dim, 1)
        ])
    
    def construct(self, x):
        attn_weights = self.attention(x)
        attn_weights = ops.Softmax(axis=0)(attn_weights)
        weighted = x * attn_weights
        return weighted.sum(axis=0)


class CLAMClassifier(ms.nn.Cell):
    def __init__(self, in_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.n_classes = n_classes
        self.attention = AttentionPool(in_dim, hidden_dim)
        self.instance_classifiers = ms.nn.CellList([
            ms.nn.SequentialCell([
                ms.nn.Dense(in_dim, hidden_dim),
                ms.nn.ReLU(),
                ms.nn.Dropout(0.5),
                ms.nn.Dense(hidden_dim, 2)
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


def extract_features_batch(model, dataset, use_amp=True):
    """批量提取特征"""
    from mindspore.dataset import GeneratorDataset
    
    ms_dataset = GeneratorDataset(
        source=dataset,
        column_names=["image", "label"],
        shuffle=False,
        num_parallel_workers=4
    )
    
    features_list = []
    labels_list = []
    
    for batch in ms_dataset:
        batch_imgs = batch[0]
        batch_labels = batch[1]
        
        if use_amp:
            with ms.amp.autocast():
                features = model(batch_imgs)
        else:
            features = model(batch_imgs)
        
        if isinstance(features, tuple):
            features = features[0]
        
        if features.ndim == 3:
            features = features.mean(axis=1)
        
        features_list.append(features.asnumpy())
        labels_list.extend(batch_labels.asnumpy())
    
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.array(labels_list)
    
    return all_features, all_labels


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  MindSpore UNI完整运行 - 任务一(速度) + 任务二(精度)")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  任务一：特征提取速度评估")
    print("=" * 60)
    
    print(f"\n[1] 加载模型...")
    model, transforms, use_amp = get_encoder(
        enc_name=args.model_name,
        device=args.device,
        test_batch=True,
        use_amp=args.use_amp,
        enable_graph_optimize=True
    )
    print(f"设备: {args.device}, AMP: {use_amp}")
    
    print(f"\n[2] 创建数据集...")
    dataset = BACHImageDataset(
        data_root=args.data_root,
        csv_file=args.csv_file,
        transforms=transforms
    )
    print(f"数据集大小: {len(dataset)}")
    
    print(f"\n[3] 提取特征 (速度测试)...")
    num_runs = 3
    times = []
    
    for run_idx in range(num_runs):
        ms.context.reset_auto_parallel_context()
        
        start_time = time.time()
        
        features, labels = extract_features_batch(
            model=model,
            dataset=dataset,
            use_amp=use_amp
        )
        
        ops.sync_all_clouds()
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        throughput = len(dataset) / elapsed
        print(f"  Run {run_idx+1}: {elapsed:.3f}s, {throughput:.2f} img/s")
    
    avg_time = sum(times) / len(times)
    avg_throughput = len(dataset) / avg_time
    wsi_per_sec = avg_throughput / 1000
    
    speed_score = wsi_per_sec * 10
    
    print(f"\n  平均速度: {avg_throughput:.2f} img/s")
    print(f"  WSI/秒: {wsi_per_sec:.4f}")
    print(f"  速度分: {speed_score:.4f}")
    
    features_path = os.path.join(args.output_dir, 'features.npy')
    labels_path = os.path.join(args.output_dir, 'labels.npy')
    np.save(features_path, features)
    np.save(labels_path, labels)
    print(f"\n特征已保存到: {args.output_dir}")
    
    print("\n" + "=" * 60)
    print("  任务二：分类精度评估")
    print("=" * 60)
    
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
    
    print(f"\n[4] 训练分类器 ({args.classifier})...")
    
    class_names = ['Normal', 'Benign', 'In situ', 'Invasive']
    
    if args.classifier == 'clam':
        ms.set_context(device_target='Ascend', mode=ms.GRAPH_MODE)
        
        train_dataset = ms.dataset.NumpySlicesDataset(
            data=(X_train.astype(np.float32), y_train.astype(np.int32)),
            column_names=["features", "labels"],
            shuffle=True
        ).batch(32, per_batch_size=32)
        
        val_dataset = ms.dataset.NumpySlicesDataset(
            data=(X_val.astype(np.float32), y_val.astype(np.int32)),
            column_names=["features", "labels"],
            shuffle=False
        ).batch(32, per_batch_size=32)
        
        test_dataset = ms.dataset.NumpySlicesDataset(
            data=(X_test.astype(np.float32), y_test.astype(np.int32)),
            column_names=["features", "labels"],
            shuffle=False
        ).batch(32, per_batch_size=32)
        
        model_cla = CLAMClassifier(
            in_dim=features.shape[1],
            n_classes=4,
            hidden_dim=128
        )
        
        loss_fn = ms.nn.CrossEntropyLoss()
        optimizer = ms.nn.Adam(model_cla.trainable_params(), learning_rate=args.clam_lr)
        train_network = ms.nn.TrainOneStepCell(loss_fn, optimizer)
        
        best_val_acc = 0
        best_model_params = None
        
        for epoch in range(args.clam_epochs):
            model_cla.set_train(True)
            total_loss = 0
            
            for batch in train_dataset:
                loss = train_network(batch[0], batch[1])
                total_loss += loss.asnumpy()
            
            def evaluate(model, dataset):
                model.set_train(False)
                all_preds = []
                all_labels = []
                for batch in dataset:
                    logits = model(batch[0])
                    preds = logits.argmax(axis=1)
                    all_preds.extend(preds.asnumpy())
                    all_labels.extend(batch[1].asnumpy())
                return accuracy_score(all_labels, all_preds)
            
            val_acc = evaluate(model_cla, val_dataset)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_params = model_cla.parameters_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.clam_epochs}, Val Acc: {val_acc:.4f}")
        
        if best_model_params:
            model_cla.load_parameters(best_model_params)
        
        model_cla.set_train(False)
        all_preds = []
        all_labels = []
        for batch in test_dataset:
            logits = model_cla(batch[0])
            preds = logits.argmax(axis=1)
            all_preds.extend(preds.asnumpy())
            all_labels.extend(batch[1].asnumpy())
        
        test_preds = np.array(all_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        test_preds = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
    
    precision_score = test_acc * 100
    
    print(f"\n  测试集 Accuracy: {test_acc:.4f}")
    print(f"  测试集 F1-Score (macro): {test_f1:.4f}")
    print(f"  精度分: {precision_score:.4f}")
    
    print("\n" + "=" * 60)
    print("  最终得分")
    print("=" * 60)
    
    final_score = speed_score * 0.5 + precision_score * 0.5
    
    print(f"速度分 (×50%): {speed_score:.4f} × 0.5 = {speed_score*0.5:.4f}")
    print(f"精度分 (×50%): {precision_score:.4f} × 0.5 = {precision_score*0.5:.4f}")
    print(f"\n综合得分: {final_score:.4f}")
    print("=" * 60)
    
    final_results = {
        'speed': {
            'avg_time_sec': avg_time,
            'avg_throughput': avg_throughput,
            'wsi_per_sec': wsi_per_sec,
            'speed_score': speed_score
        },
        'precision': {
            'accuracy': test_acc,
            'f1_macro': test_f1,
            'precision_score': precision_score
        },
        'final_score': final_score
    }
    
    results_path = os.path.join(args.output_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n结果已保存: {results_path}")
    
    return final_results


if __name__ == '__main__':
    main()
