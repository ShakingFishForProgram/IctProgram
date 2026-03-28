#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore版特征提取脚本（任务一）
==============================
功能：提取BACH数据集的特征，评估推理速度
评估指标：平均每秒提取的图像数量 (images/sec)

使用方法：
    python extract_features_ms.py --data_root ../Photos --batch_size 32
"""

import os
import sys
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

from uni.get_encoder_ms import get_encoder


class BACHImageDataset:
    """BACH图像数据集"""
    
    def __init__(self, data_root, csv_file, transforms=None):
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


def mindspore_collate_fn(batch):
    """MindSpore数据整理函数"""
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    return np.stack(images), np.array(labels)


def extract_features_batch(model, dataloader, use_amp=True):
    """批量提取特征（优化版）"""
    import pandas as pd
    
    features_list = []
    labels_list = []
    
    device = model.parameters_dict().get('device_info', {}).get('device_target', 'Ascend')
    
    for batch_imgs, batch_labels in tqdm(dataloader, desc="提取特征"):
        if isinstance(batch_imgs, np.ndarray):
            batch_tensor = Tensor.from_numpy(batch_imgs)
        else:
            batch_tensor = batch_imgs
        
        # 混合精度推理
        if use_amp:
            with ms.amp.autocast():
                features = model(batch_tensor)
        else:
            features = model(batch_tensor)
        
        if isinstance(features, tuple):
            features = features[0]
        
        if features.ndim == 3:
            features = features.mean(axis=1)
        
        features_list.append(features.asnumpy())
        labels_list.extend(batch_labels)
    
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.array(labels_list)
    
    return all_features, all_labels


def parse_args():
    parser = argparse.ArgumentParser(description='MindSpore特征提取加速评估')
    
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
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    parser.add_argument('--center_crop', action='store_true', default=True,
                       help='是否中心裁剪')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='启用混合精度')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备: auto, Ascend, GPU, CPU')
    parser.add_argument('--test_batch', action='store_true', default=True,
                       help='测试批次')
    parser.add_argument('--save_features', action='store_true', default=False,
                       help='保存特征到文件')
    parser.add_argument('--output_dir', type=str, default='./output_ms',
                       help='输出目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  MindSpore特征提取 - 任务一（速度评估）")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"批次大小: {args.batch_size}")
    print(f"混合精度: {args.use_amp}")
    print(f"数据路径: {args.data_root}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[1/3] 加载模型...")
    model, transforms, use_amp = get_encoder(
        enc_name=args.model_name,
        device=args.device,
        test_batch=args.test_batch,
        use_amp=args.use_amp,
        enable_graph_optimize=True
    )
    
    print(f"设备: {args.device}")
    print(f"AMP: {use_amp}")
    
    print("\n[2/3] 创建数据集...")
    try:
        import pandas as pd
    except:
        import pandas as pd
    
    dataset = BACHImageDataset(
        data_root=args.data_root,
        csv_file=args.csv_file,
        transforms=transforms
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    from mindspore.dataset import GeneratorDataset
    
    ms_dataset = GeneratorDataset(
        source=dataset,
        column_names=["image", "label"],
        shuffle=False,
        num_parallel_workers=args.num_workers
    )
    
    ms_dataset = ms_dataset.batch(args.batch_size, per_batch_size=args.batch_size)
    
    print("\n[3/3] 提取特征并评估速度...")
    
    num_runs = 3
    times = []
    
    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        
        ms.context.reset_auto_parallel_context()
        
        start_time = time.time()
        
        features, labels = extract_features_batch(
            model=model,
            dataloader=ms_dataset,
            use_amp=use_amp
        )
        
        ms.ops.sync_all_clouds()
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        num_images = len(features)
        throughput = num_images / elapsed
        
        print(f"图像数量: {num_images}")
        print(f"耗时: {elapsed:.3f}s")
        print(f"吞吐量: {throughput:.2f} images/sec")
    
    avg_time = sum(times) / len(times)
    avg_throughput = len(dataset) / avg_time
    
    print("\n" + "=" * 60)
    print("  速度评估结果")
    print("=" * 60)
    print(f"平均耗时: {avg_time:.3f}s")
    print(f"平均吞吐量: {avg_throughput:.2f} images/sec")
    print(f"平均每张: {avg_time/len(dataset)*1000:.2f}ms")
    print("=" * 60)
    
    if args.save_features:
        features_path = os.path.join(args.output_dir, 'features.npy')
        labels_path = os.path.join(args.output_dir, 'labels.npy')
        np.save(features_path, features)
        np.save(labels_path, labels)
        print(f"\n特征已保存:")
        print(f"  {features_path}")
        print(f"  {labels_path}")
    
    results_summary = {
        'model': args.model_name,
        'batch_size': args.batch_size,
        'use_amp': use_amp,
        'device': args.device,
        'num_images': len(dataset),
        'avg_time_sec': avg_time,
        'avg_throughput': avg_throughput,
    }
    
    import json
    summary_path = os.path.join(args.output_dir, 'speed_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n结果已保存: {summary_path}")
    
    return results_summary


if __name__ == '__main__':
    main()
