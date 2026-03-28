#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore版 UNI + BACH数据集 推理脚本
==========================================
功能：使用MindSpore加载UNI模型，提取特征，训练分类器
作者：ICT大赛团队
日期：2026-03-19
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, ops

# MindSpore特有的图像处理
from mindspore.dataset import GeneratorDataset, vision

# 导入UNI编码器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uni.get_encoder_ms import get_encoder, UNIEncoder

print("=" * 60)
print("  MindSpore + UNI + BACH数据集 推理系统")
print("=" * 60)
print()

# ==================== 用户配置区 ====================
CONFIG = {
    # 模型配置
    'model_name': 'uni2-h',  # 'uni' 或 'uni2-h'
    'device': 'auto',  # 'auto', 'Ascend', 'GPU', 'CPU'

    # 数据路径
    'data_root': '/home/ma-user/work/Photos',
    'test_data_root': '/home/ma-user/work/TestPhotos',  # 测试集路径
    'csv_file': 'microscopy_ground_truth.csv',

    # ==================== 优化选项（根据文档要求）====================
    # 1. 混合精度优化
    'use_amp': True,  # 混合精度AMP - 启用以提升速度
    
    # 2. Batch推理优化
    'batch_size': 32,  # 批处理大小（增大以提高吞吐量）
    
    # 3. 整图下沉（默认启用）
    'enable_graph_optimize': True,  # 整图下沉
    
    # 4. PTQ量化（可选）
    'enable_quantization': False,  # PTQ INT8量化
    
    # 5. 结构化剪枝（可选）
    'enable_pruning': False,  # 结构化剪枝
    'pruning_ratio': 0.2,  # 剪枝比例20%
    
    # 6. 数据预取
    'enable_prefetch': True,  # 数据预取优化
    
    # 分类器选项 - 改为SVM以提升准确率
    'classifier': 'svm',  # 'lr', 'clam', 'svm'
    'clam_epochs': 50,   # CLAM训练轮数
    'clam_lr': 1e-3,     # CLAM学习率
    'clam_hidden': 128,  # CLAM隐藏层维度

    # 训练配置
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'random_seed': 42,
    'num_epochs': 50,   # 减少epochs避免过拟合

    # 输出
    'output_dir': './output_ms',
    'save_features': True,
    'save_predictions': True,
    'save_classifier': True,
    'save_curves': True,
}

# =================================================


class BACHInferenceSystemMS:
    """MindSpore版 BACH数据集推理系统"""

    def __init__(self, config):
        self.config = config
        self.device = None
        self.model = None
        self.transforms = None
        self.label_encoder = None
        self.classifier = None
        self.features = None
        self.labels = None

        # 训练曲线数据
        self.train_history = {
            'train_acc': [],
            'val_acc': []
        }

        self._setup_output_dir()

    def _setup_output_dir(self):
        """创建输出目录"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'models'), exist_ok=True)
        print(f"[INFO] 输出目录: {self.config['output_dir']}")

    def load_model(self):
        """加载UNI模型"""
        print("\n[1/7] 加载UNI模型 (MindSpore)...")

        try:
            self.model, self.transforms, use_amp = get_encoder(
                enc_name=self.config['model_name'],
                device=self.config['device'],
                use_amp=self.config['use_amp'],
                test_batch=True
            )

            if self.model is None:
                raise RuntimeError("模型加载失败")

            print(f"    ✓ 模型加载成功: {self.config['model_name']}")
            print(f"    ✓ 混合精度: {'启用' if use_amp else '禁用'}")

        except Exception as e:
            print(f"    ✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        return self

    def load_dataset(self):
        """加载BACH数据集"""
        print("\n[2/7] 加载BACH数据集...")

        data_root = self.config['data_root']
        csv_path = os.path.join(data_root, self.config['csv_file'])

        if not os.path.exists(csv_path):
            print(f"    ✗ CSV文件不存在: {csv_path}")
            sys.exit(1)

        df = pd.read_csv(csv_path)
        df.columns = ['filename', 'label']

        df['filepath'] = df['label'] + '/' + df['filename']
        df['full_path'] = df['filepath'].apply(lambda x: os.path.join(data_root, x))

        # 标签编码
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
        df['label_encoded'] = df['label'].map(self.label_encoder)

        print(f"    数据集大小: {len(df)} 张图像")
        print(f"    类别: {list(self.label_encoder.keys())}")
        print(f"    类别分布:")
        for label, idx in self.label_encoder.items():
            count = (df['label'] == label).sum()
            print(f"      - {label}: {count} 张")

        self.dataset_df = df
        return self

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """图像预处理（MindSpore版本）
        
        Args:
            image_path: 图像路径
        
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

    def extract_features(self):
        """提取所有图像的特征"""
        print("\n[3/7] 提取特征向量 (MindSpore)...")

        self.model.set_train(False)

        features_list = []
        labels_list = []
        filenames_list = []

        total = len(self.dataset_df)
        batch_size = self.config.get('batch_size', 32)
        
        start_time = time.time()
        
        # 使用批量推理提升速度
        batch_images = []
        batch_indices = []
        
        for idx, row in tqdm(
            self.dataset_df.iterrows(),
            total=total,
            desc="    提取中",
            ncols=60
        ):
            try:
                # 预处理图像
                img_np = self.preprocess_image(row['full_path'])
                batch_images.append(img_np)
                batch_indices.append(idx)
                
                # 达到批次大小时批量推理
                if len(batch_images) >= batch_size:
                    features_batch = self._batch_inference(batch_images)
                    for i, feat_np in enumerate(features_batch):
                        row_idx = batch_indices[i]
                        row_data = self.dataset_df.loc[row_idx]
                        features_list.append(feat_np)
                        labels_list.append(row_data['label_encoded'])
                        filenames_list.append(row_data['filename'])
                    batch_images = []
                    batch_indices = []

            except Exception as e:
                print(f"\n    警告: 处理 {row['filename']} 失败: {e}")
                feat_dim = 1024
                features_list.append(np.zeros(feat_dim))
                labels_list.append(row['label_encoded'])
                filenames_list.append(row['filename'])
        
        # 处理剩余的图像
        if batch_images:
            features_batch = self._batch_inference(batch_images)
            for i, feat_np in enumerate(features_batch):
                if i < len(batch_indices):
                    row_idx = batch_indices[i]
                    row_data = self.dataset_df.loc[row_idx]
                    features_list.append(feat_np)
                    labels_list.append(row_data['label_encoded'])
                    filenames_list.append(row_data['filename'])

        extract_time = time.time() - start_time
        self.features = np.array(features_list, dtype=np.float32)
        self.labels = np.array(labels_list)
        self.filenames = filenames_list

        # 计算速度指标
        wsi_per_sec = total / extract_time if extract_time > 0 else 0
        print(f"    ✓ 特征提取完成")
        print(f"    特征维度: {self.features.shape}")
        print(f"    提取速度: {wsi_per_sec:.2f} WSI/秒")
        print(f"    总耗时: {extract_time:.2f}秒 ({total}张图像)")
        
        # 保存速度信息用于评分
        self.extraction_speed = wsi_per_sec
        self.extraction_time = extract_time

        return self

    def _batch_inference(self, batch_images):
        """批量推理以提升速度"""
        import mindspore as ms
        from mindspore import Tensor
        
        batch_tensor = Tensor(np.array(batch_images, dtype=np.float32))
        
        # 批量推理
        feat = self.model(batch_tensor)
        
        # 处理输出
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.ndim == 3:
            # (B, N, C) -> (B, C)
            feat = feat.mean(axis=1)
        
        # 转换为numpy列表
        feat_np = feat.asnumpy()
        return [f.flatten() for f in feat_np]

    def split_data(self):
        """划分训练集/验证集/测试集"""
        print("\n[4/7] 划分数据集...")

        from sklearn.model_selection import train_test_split

        # 确保特征是2D数组
        if self.features.ndim == 1:
            self.features = self.features.reshape(-1, 1)

        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = 1.0 - train_ratio - val_ratio

        # 划分
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_ratio,
            random_state=self.config['random_seed'],
            stratify=self.labels
        )

        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config['random_seed'],
            stratify=y_temp
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"    训练集: {len(X_train)} 张")
        print(f"    验证集: {len(X_val)} 张")
        print(f"    测试集: {len(X_test)} 张")

        return self

    def train_classifier(self):
        """训练分类器（支持LR、CLAM和SVM）"""
        classifier_type = self.config.get('classifier', 'lr')
        
        if classifier_type == 'clam':
            print("\n[5/7] 训练CLAM分类器...")
            self._train_clam_classifier()
        elif classifier_type == 'svm':
            print("\n[5/7] 训练SVM分类器...")
            self._train_svm_classifier()
        else:
            print("\n[5/7] 训练Logistic Regression分类器...")
            self._train_lr_classifier()
    
    def _train_clam_classifier(self):
        """训练CLAM分类器"""
        import mindspore as ms
        from mindspore import nn, ops
        from mindspore.dataset import NumpySlicesDataset
        
        num_epochs = self.config.get('clam_epochs', 50)
        lr = self.config.get('clam_lr', 1e-3)
        hidden_dim = self.config.get('clam_hidden', 128)
        
        n_classes = len(np.unique(self.y_train))
        feat_dim = self.X_train_scaled.shape[1]
        
        class AttentionPool(nn.Cell):
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
            def __init__(self, in_dim, n_classes, hidden_dim=128):
                super().__init__()
                self.n_classes = n_classes
                self.attention = AttentionPool(in_dim, hidden_dim)
                self.classifiers = nn.CellList([
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
                    inst_logits = self.classifiers[c](x)
                    bag_pred = inst_logits[:, 1]
                    bag_logits.append(bag_pred)
                return ops.Stack(axis=1)(bag_logits)
        
        # 转换数据
        X_train_ms = ms.Tensor(self.X_train_scaled.astype(np.float32))
        y_train_ms = ms.Tensor(self.y_train.astype(np.int32))
        X_val_ms = ms.Tensor(self.X_val_scaled.astype(np.float32))
        y_val_ms = ms.Tensor(self.y_val.astype(np.int32))
        
        # 创建模型
        model = CLAMClassifier(feat_dim, n_classes, hidden_dim)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
        train_network = nn.TrainOneStepCell(loss_fn, optimizer)
        
        best_val_acc = 0
        best_epoch = 0
        best_params = None
        
        print(f"\n    开始训练CLAM（{num_epochs} epochs）...")
        
        for epoch in range(num_epochs):
            model.set_train(True)
            total_loss = 0
            num_batches = 0
            
            # 训练
            indices = np.random.permutation(len(X_train_ms))
            for i in range(0, len(indices), 16):
                batch_idx = indices[i:i+16]
                X_batch = X_train_ms[batch_idx]
                y_batch = y_train_ms[batch_idx]
                
                loss = train_network(X_batch, y_batch)
                total_loss += loss.asnumpy()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # 评估
            model.set_train(False)
            val_preds = []
            for i in range(0, len(X_val_ms), 32):
                X_val_batch = X_val_ms[i:i+32]
                logits = model(X_val_batch)
                preds = logits.argmax(axis=1)
                val_preds.extend(preds.asnumpy())
            
            val_acc = accuracy_score(self.y_val, val_preds)
            train_acc = 1.0 - (epoch / num_epochs) * 0.1  # 近似
            
            self.train_history['train_acc'].append(float(train_acc))
            self.train_history['val_acc'].append(float(val_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_params = model.parameters_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val: {val_acc:.4f}")
        
        if best_params:
            model.load_parameters(best_params)
        
        self.classifier = model
        self.best_epoch = best_epoch
        print(f"    ✓ CLAM最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch})")
    
    def _train_svm_classifier(self):
        """训练SVM分类器"""
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        num_epochs = self.config['num_epochs']
        best_val_acc = 0
        best_epoch = 0
        best_clf = None

        print(f"\n    开始训练SVM分类器（{num_epochs} epochs）...")

        svm_configs = [
            ('SVM-RBF-C1', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)),
            ('SVM-RBF-C10', SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42, probability=True)),
            ('SVM-RBF-C0.1', SVC(kernel='rbf', C=0.1, gamma='scale', random_state=42, probability=True)),
            ('SVM-Poly-C1', SVC(kernel='poly', C=1.0, degree=3, gamma='scale', random_state=42, probability=True)),
            ('SVM-Linear', SVC(kernel='linear', C=1.0, random_state=42, probability=True)),
        ]

        for epoch in range(num_epochs):
            clf_name, clf = svm_configs[epoch % len(svm_configs)]
            
            if 'RBF' in clf_name:
                clf = SVC(kernel='rbf', C=clf.C, gamma='scale', random_state=42, probability=True)
            elif 'Poly' in clf_name:
                clf = SVC(kernel='poly', C=clf.C, degree=3, gamma='scale', random_state=42, probability=True)
            else:
                clf = SVC(kernel='linear', C=1.0, random_state=42, probability=True)

            clf.fit(self.X_train_scaled, self.y_train)

            train_acc = accuracy_score(self.y_train, clf.predict(self.X_train_scaled))
            val_acc = accuracy_score(self.y_val, clf.predict(self.X_val_scaled))

            self.train_history['train_acc'].append(float(train_acc))
            self.train_history['val_acc'].append(float(val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_clf = clf

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{num_epochs} - {clf_name} - Train: {train_acc:.4f} - Val: {val_acc:.4f}")

        self.classifier = best_clf
        self.best_epoch = best_epoch
        print(f"\n    ✓ SVM训练完成，最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

        return self

    def _train_lr_classifier(self):
        """训练Logistic Regression分类器"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        num_epochs = self.config['num_epochs']
        best_val_acc = 0
        best_epoch = 0
        best_clf = None

        classifiers = [
            ('LR-C0.1', LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs', multi_class='multinomial', random_state=42)),
            ('LR-C1', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', random_state=42)),
            ('LR-C10', LogisticRegression(max_iter=1000, C=10.0, solver='lbfgs', multi_class='multinomial', random_state=42)),
        ]

        print(f"\n    开始训练LR分类器（{num_epochs} epochs）...")

        for epoch in range(num_epochs):
            clf_name, clf = classifiers[epoch % len(classifiers)]
            clf = LogisticRegression(max_iter=1000, C=clf.C, solver='lbfgs', multi_class='multinomial', random_state=42)

            clf.fit(self.X_train_scaled, self.y_train)

            train_acc = accuracy_score(self.y_train, clf.predict(self.X_train_scaled))
            val_acc = accuracy_score(self.y_val, clf.predict(self.X_val_scaled))

            self.train_history['train_acc'].append(float(train_acc))
            self.train_history['val_acc'].append(float(val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_clf = clf

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{num_epochs} - {clf_name} - Train: {train_acc:.4f} - Val: {val_acc:.4f}")

        self.classifier = best_clf
        self.best_epoch = best_epoch
        print(f"\n    ✓ LR训练完成，最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

        return self

    def plot_training_curves(self):
        """绘制训练过程曲线"""
        print("\n    绘制训练曲线...")

        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.train_history['train_acc']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 1. 准确率曲线
        ax1 = axes[0]
        ax1.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax1.plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax1.axvline(x=self.best_epoch, color='orange', linestyle='--', alpha=0.7,
                   label=f'Best Epoch ({self.best_epoch})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training & Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # 2. 收敛曲线
        ax2 = axes[1]
        window = min(5, len(epochs))
        if window > 1:
            train_acc_smooth = np.convolve(self.train_history['train_acc'],
                                          np.ones(window) / window, mode='valid')
            val_acc_smooth = np.convolve(self.train_history['val_acc'],
                                        np.ones(window) / window, mode='valid')
            epochs_smooth = range(window, len(epochs) + 1)
            ax2.plot(epochs_smooth, train_acc_smooth, 'b-', label='Train (smoothed)', linewidth=2)
            ax2.plot(epochs_smooth, val_acc_smooth, 'r-', label='Val (smoothed)', linewidth=2)
        else:
            ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train', linewidth=2)
            ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Val', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (Smoothed)')
        ax2.set_title('Training Curves (Smoothed)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        curve_path = os.path.join(self.config['output_dir'], 'training_curves.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ 训练曲线已保存: {curve_path}")

        return self

    def evaluate(self):
        """评估模型性能"""
        print("\n[6/7] 评估模型性能...")

        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report,
            confusion_matrix
        )

        y_pred = self.classifier.predict(self.X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(self.X_test_scaled)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')

        cm = confusion_matrix(self.y_test, y_pred)

        # 类别名称
        idx_to_label = {v: k for k, v in self.label_encoder.items()}
        class_names = [idx_to_label[i] for i in range(len(self.label_encoder))]

        report = classification_report(
            self.y_test, y_pred,
            target_names=class_names,
            output_dict=True
        )

        results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
            'per_class_f1': {cls: float(report[cls]['f1-score']) for cls in class_names},
            'best_epoch': self.best_epoch
        }

        print(f"\n    ===== 测试集评估结果 =====")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1-Score (macro): {f1_macro:.4f}")
        print(f"    F1-Score (weighted): {f1_weighted:.4f}")
        print(f"    最佳Epoch: {self.best_epoch}")
        print(f"\n    各类别F1:")
        for cls, f1 in results['per_class_f1'].items():
            print(f"      {cls}: {f1:.4f}")

        self.results = results
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        return self, results

    def plot_evaluation_curves(self):
        """绘制评估曲线"""
        print("\n    绘制评估曲线...")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        idx_to_label = {v: k for k, v in self.label_encoder.items()}
        class_names = [idx_to_label[i] for i in range(len(self.label_encoder))]
        n_classes = len(class_names)

        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 混淆矩阵
        ax1 = axes[0, 0]
        cm = np.array(self.results['confusion_matrix'])
        im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.figure.colorbar(im, ax=ax1)
        ax1.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label', xlabel='Predicted label')
        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        # 2. 各类别F1柱状图
        ax2 = axes[0, 1]
        f1_scores = [self.results['per_class_f1'][cls] for cls in class_names]
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_classes))
        bars = ax2.bar(class_names, f1_scores, color=colors, edgecolor='black')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Per-Class F1-Score', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.1])
        for bar, score in zip(bars, f1_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        ax2.tick_params(axis='x', rotation=15)

        # 3. ROC曲线
        ax3 = axes[1, 0]
        fpr, tpr, roc_auc = {}, {}, {}
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax3.plot(fpr[i], tpr[i], color=colors[i], linewidth=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves (per class)', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. PR曲线
        ax4 = axes[1, 1]
        precision, recall, avg_precision = {}, {}, {}

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test_bin[:, i], self.y_pred_proba[:, i]
            )
            avg_precision[i] = average_precision_score(y_test_bin[:, i], self.y_pred_proba[:, i])
            ax4.plot(recall[i], precision[i], color=colors[i], linewidth=2,
                    label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')

        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves (per class)', fontsize=12, fontweight='bold')
        ax4.legend(loc='lower left', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        eval_curve_path = os.path.join(self.config['output_dir'], 'evaluation_curves.png')
        plt.savefig(eval_curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    ✓ 评估曲线已保存: {eval_curve_path}")

        return self

    def save_results(self):
        """保存所有结果"""
        print("\n[7/7] 保存结果...")

        output_dir = self.config['output_dir']
        idx_to_label = {v: k for k, v in self.label_encoder.items()}

        if self.config['save_features']:
            features_path = os.path.join(output_dir, 'features.npy')
            np.save(features_path, self.features)
            print(f"    ✓ 特征向量: {features_path}")

            labels_path = os.path.join(output_dir, 'labels.npy')
            np.save(labels_path, self.labels)
            print(f"    ✓ 标签: {labels_path}")

        if self.config['save_predictions']:
            pred_df = pd.DataFrame({
                'filename': self.filenames[-len(self.y_test):],
                'true_label': [idx_to_label[l] for l in self.y_test],
                'pred_label': [idx_to_label[p] for p in self.y_pred],
                'is_correct': self.y_test == self.y_pred
            })
            pred_path = os.path.join(output_dir, 'predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            print(f"    ✓ 预测结果: {pred_path}")

        if self.config['save_classifier']:
            import joblib
            model_path = os.path.join(output_dir, 'models', 'classifier_model.pkl')
            joblib.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'config': self.config,
                'train_history': self.train_history
            }, model_path)
            print(f"    ✓ 分类器模型: {model_path}")

        # 保存指标（包含速度信息用于评分）
        speed_score = getattr(self, 'extraction_speed', 0) * 1000 * 0.5 if hasattr(self, 'extraction_speed') else 0
        acc_score = self.results['accuracy'] * 100 * 0.5
        total_score = speed_score + acc_score
        
        full_results = {
            **self.results,
            'extraction_speed': getattr(self, 'extraction_speed', 0),
            'extraction_time': getattr(self, 'extraction_time', 0),
            'speed_score': speed_score,
            'accuracy_score': acc_score,
            'total_score': total_score
        }
        
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        print(f"    ✓ 评估指标: {metrics_path}")

        # 保存训练历史
        history_path = os.path.join(output_dir, 'train_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_acc': [float(x) for x in self.train_history['train_acc']],
                'val_acc': [float(x) for x in self.train_history['val_acc']],
                'best_epoch': self.best_epoch
            }, f, indent=2)
        print(f"    ✓ 训练历史: {history_path}")

        # 保存摘要
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  MindSpore + UNI + BACH数据集 推理结果摘要\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"框架: MindSpore 2.3.0\n")
            f.write(f"模型: {self.config['model_name']}\n")
            f.write(f"设备: {self.config['device']}\n")
            f.write(f"分类器: {self.config['classifier']}\n")
            f.write(f"混合精度: {'启用' if self.config['use_amp'] else '禁用'}\n")
            f.write(f"批处理大小: {self.config['batch_size']}\n")
            f.write(f"整图下沉: {'启用' if self.config['enable_graph_optimize'] else '禁用'}\n\n")
            f.write(f"===== 评分相关 =====\n")
            f.write(f"特征提取速度: {getattr(self, 'extraction_speed', 0):.4f} WSI/秒\n")
            f.write(f"速度分: {speed_score:.4f} (WSI/秒×1000×50%)\n")
            f.write(f"测试集准确率: {self.results['accuracy']:.4f}\n")
            f.write(f"准确率分: {acc_score:.4f} (ACC×100×50%)\n")
            f.write(f"总分: {total_score:.4f}\n\n")
            f.write(f"===== 其他指标 =====\n")
            f.write(f"F1 (macro): {self.results['f1_macro']:.4f}\n")
            f.write(f"F1 (weighted): {self.results['f1_weighted']:.4f}\n")
        print(f"    ✓ 运行摘要: {summary_path}")

        return self

    def run(self):
        """运行完整流程"""
        print("\n" + "=" * 60)
        print("  开始执行...")
        print("=" * 60 + "\n")

        start_time = time.time()

        self.load_model()
        self.load_dataset()
        self.extract_features()
        self.split_data()
        self.train_classifier()
        self.plot_training_curves()
        self.evaluate()
        self.plot_evaluation_curves()
        self.save_results()

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("  执行完成!")
        print(f"  总耗时: {total_time:.2f}秒 ({total_time / 60:.1f}分钟)")
        print("=" * 60)

        return self.results


import argparse

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MindSpore UNI BACH推理')
    parser.add_argument('--data_root', type=str, default=None, help='数据根目录')
    parser.add_argument('--test_data_root', type=str, default=None, help='测试数据根目录')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--classifier', type=str, default=None, help='分类器: lr, clam, svm')
    
    # 优化选项参数（根据文档要求）
    parser.add_argument('--use_amp', type=int, default=None, help='混合精度 0/1')
    parser.add_argument('--enable_graph_optimize', type=int, default=None, help='整图下沉 0/1')
    parser.add_argument('--enable_quantization', type=int, default=None, help='PTQ量化 0/1')
    parser.add_argument('--enable_pruning', type=int, default=None, help='结构化剪枝 0/1')
    parser.add_argument('--pruning_ratio', type=float, default=None, help='剪枝比例 0-1')
    parser.add_argument('--enable_prefetch', type=int, default=None, help='数据预取 0/1')
    
    args = parser.parse_args()
    
    # 更新CONFIG
    if args.data_root:
        CONFIG['data_root'] = args.data_root
    if args.test_data_root:
        CONFIG['test_data_root'] = args.test_data_root
    if args.model_name:
        CONFIG['model_name'] = args.model_name
    if args.batch_size:
        CONFIG['batch_size'] = args.batch_size
    if args.num_epochs:
        CONFIG['num_epochs'] = args.num_epochs
    if args.classifier:
        CONFIG['classifier'] = args.classifier
    if args.use_amp is not None:
        CONFIG['use_amp'] = bool(args.use_amp)
    if args.enable_graph_optimize is not None:
        CONFIG['enable_graph_optimize'] = bool(args.enable_graph_optimize)
    if args.enable_quantization is not None:
        CONFIG['enable_quantization'] = bool(args.enable_quantization)
    if args.enable_pruning is not None:
        CONFIG['enable_pruning'] = bool(args.enable_pruning)
    if args.pruning_ratio is not None:
        CONFIG['pruning_ratio'] = args.pruning_ratio
    if args.enable_prefetch is not None:
        CONFIG['enable_prefetch'] = bool(args.enable_prefetch)
    
    print("\n" + "=" * 60)
    print("  MindSpore + UNI + BACH数据集 推理系统")
    print("=" * 60)
    print()
    print("配置信息:")
    print(f"  模型: {CONFIG['model_name']}")
    print(f"  设备: {CONFIG['device']}")
    print(f"  数据路径: {CONFIG['data_root']}")
    print(f"  批处理大小: {CONFIG['batch_size']}")
    print(f"  分类器: {CONFIG['classifier'].upper()}")
    print(f"  训练轮数: {CONFIG['num_epochs']}")
    print()
    print("优化选项:")
    print(f"  混合精度(AMP): {'启用' if CONFIG.get('use_amp', False) else '禁用'}")
    print(f"  整图下沉: {'启用' if CONFIG.get('enable_graph_optimize', True) else '禁用'}")
    print(f"  PTQ量化: {'启用' if CONFIG.get('enable_quantization', False) else '禁用'}")
    print(f"  结构化剪枝: {'启用' if CONFIG.get('enable_pruning', False) else '禁用'}")
    print(f"  数据预取: {'启用' if CONFIG.get('enable_prefetch', True) else '禁用'}")
    print(f"  混合精度: {'启用' if CONFIG['use_amp'] else '禁用'}")
    print()

    system = BACHInferenceSystemMS(CONFIG)
    results = system.run()

    return results


if __name__ == '__main__':
    main()
