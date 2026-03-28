#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务二：推理脚本
==============
使用训练好的SVM分类器进行推理

用于：评委测试或其他数据集验证
"""

import os
import sys
import json
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uni.get_encoder_ms import get_encoder


CONFIG = {
    'data_dir': '/home/ma-user/work/Photos',
    'output_dir': './output_task2',
    'model_name': 'uni2-h',
    'device': 'auto',
    'use_amp': True,
    'enable_graph_optimize': True,
    'batch_size': 32,
    'img_size': 224,
}


def parse_args():
    parser = argparse.ArgumentParser(description='任务二：推理')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output_task2', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    return parser.parse_args()


def main():
    global CONFIG
    
    args = parse_args()
    
    if args.data_dir:
        CONFIG['data_dir'] = args.data_dir
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    if args.batch_size:
        CONFIG['batch_size'] = args.batch_size
    
    # 模型路径
    model_path = os.path.join(CONFIG['output_dir'], 'classifier_model.pkl')
    model_info_path = os.path.join(CONFIG['output_dir'], 'model_info.json')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    if not os.path.exists(model_info_path):
        print(f"错误: 模型信息文件不存在 {model_info_path}")
        return
    
    # 加载UNI2模型
    print("\n[1/3] 加载UNI2模型...")
    encoder, _, _ = get_encoder(
        enc_name=CONFIG['model_name'],
        device=CONFIG['device'],
        use_amp=CONFIG['use_amp'],
        test_batch=True,
        enable_graph_optimize=CONFIG['enable_graph_optimize']
    )
    print(f"    ✓ 模型加载成功")
    
    # 加载SVM分类器
    print("\n[2/3] 加载SVM分类器...")
    clf = joblib.load(model_path)
    
    # 加载模型信息
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    label_encoder = model_info['label_encoder']
    scaler_mean = np.array(model_info['scaler_mean'])
    scaler_scale = np.array(model_info['scaler_scale'])
    
    print(f"    ✓ 分类器加载成功")
    print(f"    ✓ 类别: {list(label_encoder.keys())}")
    
    # 标准化器
    class Scaler:
        def __init__(self, mean, scale):
            self.mean_ = mean
            self.scale_ = scale
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_
    
    scaler = Scaler(scaler_mean, scaler_scale)
    
    # 预处理图像
    def preprocess_image(image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((CONFIG['img_size'], CONFIG['img_size']))
        
        img_np = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        
        return img_np.astype(np.float32)
    
    # 特征提取
    def extract_features(image_paths, encoder, batch_size=32):
        encoder.set_train(False)
        
        features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="    提取特征"):
            batch = []
            for path in image_paths[i:i+batch_size]:
                img = preprocess_image(path)
                batch.append(img)
            
            batch_tensor = Tensor(np.array(batch, dtype=np.float32))
            feat = encoder(batch_tensor)
            
            if isinstance(feat, tuple):
                feat = feat[0]
            if feat.ndim == 3:
                feat = feat.mean(axis=1)
            
            features.extend(feat.asnumpy())
        
        return np.array(features)
    
    # 查找图像
    image_paths = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(sorted(glob.glob(os.path.join(CONFIG['data_dir'], '*', ext))))
        image_paths.extend(sorted(glob.glob(os.path.join(CONFIG['data_dir'], ext))))
    
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"    错误: 未找到图像文件")
        return
    
    # 处理图像
    print("\n[3/3] 处理图像...")
    print(f"    找到 {len(image_paths)} 个图像")
    
    # 提取特征
    features = extract_features(image_paths, encoder, CONFIG['batch_size'])
    
    # 标准化
    features_scaled = scaler.transform(features)
    
    # 分类预测
    preds = clf.predict(features_scaled)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("  分类结果")
    print("=" * 50)
    
    idx_to_label = {v: k for k, v in label_encoder.items()}
    
    results = []
    for path, pred in zip(image_paths, preds):
        filename = os.path.basename(path)
        label = idx_to_label[int(pred)]
        results.append({'filename': filename, 'prediction': label})
        print(f"  {filename}: {label}")
    
    # 统计类别分布
    pred_counts = {}
    for pred in preds:
        label = idx_to_label[int(pred)]
        pred_counts[label] = pred_counts.get(label, 0) + 1
    
    print("\n  类别分布:")
    for label, count in pred_counts.items():
        print(f"    {label}: {count}")
    
    # 保存结果
    result_path = os.path.join(CONFIG['output_dir'], 'inference_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n    结果已保存: {result_path}")
    print("=" * 50)


if __name__ == '__main__':
    import argparse
    main()