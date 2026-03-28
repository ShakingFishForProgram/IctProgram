#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore版 UNI 分类器预测脚本
==========================================
功能：加载已训练好的分类器，对新图像进行预测
作者：ICT大赛团队
日期：2026-03-19
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import Tensor

# 导入UNI编码器
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uni.get_encoder_ms import get_encoder

import joblib


def load_trained_system(model_path='./output_ms/models/classifier_model.pkl'):
    """加载已训练好的系统和模型"""
    
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 run_bach_inference_ms.py 训练模型")
        sys.exit(1)
        
    print(f"加载分类器: {model_path}")
    data = joblib.load(model_path)
    
    classifier = data['classifier']
    label_encoder = data['label_encoder']
    config = data['config']
    
    print(f"类别: {list(label_encoder.keys())}")
    
    # 加载MindSpore编码器
    print("加载UNI特征提取器 (MindSpore)...")
    uni_model, transforms, use_amp = get_encoder(
        enc_name=config['model_name'],
        device='auto',
        test_batch=True
    )
    
    return {
        'classifier': classifier,
        'label_encoder': label_encoder,
        'uni_model': uni_model,
        'transforms': transforms,
        'config': config
    }


def preprocess_image(image_path: str, transforms: list = None) -> np.ndarray:
    """图像预处理
    
    Args:
        image_path: 图像路径
        transforms: 预处理变换列表
    
    Returns:
        预处理后的图像数组 (C, H, W)
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image = image.resize((224, 224))
    
    # 转换为numpy数组
    img_np = np.asarray(image, dtype=np.float32) / 255.0
    
    # 归一化 (ImageNet参数)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # 转换维度 (H, W, C) -> (C, H, W)
    img_np = img_np.transpose(2, 0, 1)
    
    return img_np.astype(np.float32)


def predict_image(system, image_path):
    """对单张图像进行预测"""
    
    classifier = system['classifier']
    label_encoder = system['label_encoder']
    uni_model = system['uni_model']
    config = system['config']
    
    if not os.path.exists(image_path):
        print(f"[错误] 图像不存在: {image_path}")
        return None
        
    # 预处理图像
    img_np = preprocess_image(image_path)
    
    # 转换为MindSpore张量
    img_tensor = Tensor.from_numpy(img_np).expand_dims(0)
    
    # 提取特征
    uni_model.set_train(False)
    features = uni_model(img_tensor)
    
    # 处理输出
    if isinstance(features, tuple):
        features = features[0]
    if features.ndim == 3:
        features = features.mean(axis=1)
    
    # 转换为numpy
    features_np = features.asnumpy().reshape(1, -1)
    
    # 预测
    pred_label = classifier.predict(features_np)[0]
    pred_proba = classifier.predict_proba(features_np)[0]
    
    # 转换标签
    idx_to_label = {v: k for k, v in label_encoder.items()}
    label_name = idx_to_label[pred_label]
    confidence = pred_proba[pred_label]
    
    return {
        'label': label_name,
        'confidence': float(confidence),
        'probabilities': {
            idx_to_label[i]: float(prob) 
            for i, prob in enumerate(pred_proba)
        }
    }


def predict_batch(system, image_dir, extensions=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
    """批量预测文件夹中的图像"""
    
    classifier = system['classifier']
    label_encoder = system['label_encoder']
    uni_model = system['uni_model']
    config = system['config']
    
    idx_to_label = {v: k for k, v in label_encoder.items()}
    
    # 收集图像文件
    image_files = []
    for ext in extensions:
        image_files.extend([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(ext)
        ])
    
    if not image_files:
        print(f"[警告] 未找到图像文件: {image_dir}")
        return []
        
    print(f"找到 {len(image_files)} 张图像")
    
    # 设置模型为推理模式
    uni_model.set_train(False)
    
    results = []
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        
        # 预处理
        img_np = preprocess_image(image_path)
        img_tensor = Tensor.from_numpy(img_np).expand_dims(0)
        
        # 提取特征
        features = uni_model(img_tensor)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim == 3:
            features = features.mean(axis=1)
        features_np = features.asnumpy().reshape(1, -1)
        
        # 预测
        pred_label = classifier.predict(features_np)[0]
        pred_proba = classifier.predict_proba(features_np)[0]
        
        label_name = idx_to_label[pred_label]
        confidence = pred_proba[pred_label]
        
        results.append({
            'filename': filename,
            'label': label_name,
            'confidence': float(confidence)
        })
        
    return results


def print_prediction(result):
    """打印预测结果"""
    print("\n" + "=" * 40)
    print("  预测结果")
    print("=" * 40)
    print(f"  类别: {result['label']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print("\n  各类别概率:")
    for cls, prob in result['probabilities'].items():
        bar = '█' * int(prob * 20)
        print(f"    {cls:15s} {prob:.2%} {bar}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description='UNI分类器预测 (MindSpore版)')
    parser.add_argument('--model', '-m', 
                       default='./output_ms/models/classifier_model.pkl',
                       help='分类器模型路径')
    parser.add_argument('--image', '-i',
                       help='单张图像路径')
    parser.add_argument('--dir', '-d',
                       help='图像文件夹路径')
    parser.add_argument('--output', '-o',
                       help='输出CSV路径')
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        print("请指定图像路径或文件夹路径")
        print("用法:")
        print("  python load_and_predict_ms.py -i image.tif")
        print("  python load_and_predict_ms.py -d ./images/")
        sys.exit(1)
        
    system = load_trained_system(args.model)
    
    if args.image:
        result = predict_image(system, args.image)
        if result:
            print_prediction(result)
            
    elif args.dir:
        results = predict_batch(system, args.dir)
        
        print(f"\n预测完成，共 {len(results)} 张图像")
        
        for r in results:
            print(f"  {r['filename']:30s} -> {r['label']:10s} ({r['confidence']:.2%})")
            
        if args.output:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
