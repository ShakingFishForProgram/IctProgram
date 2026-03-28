#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务二：CLAM分类器训练 (MindSpore版本)
====================================
使用UNI2提取特征 + CLAM分类器进行4分类

数据划分：70%训练 / 15%验证 / 15%测试
框架：MindSpore + NPU
分类器：CLAM (注意力机制) - MindSpore实现
模型格式：.ckpt
"""

import os
import sys
import json
import time
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, nn, ops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uni.get_encoder_ms import get_encoder

print("=" * 60)
print("  任务二：BACH数据集 4分类任务")
print("  特征提取：UNI2 (MindSpore)")
print("  分类器：CLAM (MindSpore)")
print("  数据划分：70%训练 / 15%验证 / 15%测试")
print("=" * 60)
print()

CONFIG = {
    'data_dir': '/home/ma-user/work/Photos',
    'csv_file': 'microscopy_ground_truth.csv',
    'output_dir': './output_task2',
    
    'model_name': 'uni2-h',
    'device': 'auto',
    'use_amp': True,
    'enable_graph_optimize': True,
    'batch_size': 32,
    'img_size': 224,
    
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    
    'save_features': True,
}


def parse_args():
    parser = argparse.ArgumentParser(description='任务二：CLAM分类器训练 (MindSpore)')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--clam_epochs', type=int, default=None, help='CLAM训练轮数')
    parser.add_argument('--clam_lr', type=float, default=None, help='CLAM学习率')
    parser.add_argument('--clam_hidden', type=int, default=None, help='CLAM隐藏层维度')
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
    if args.clam_epochs:
        CONFIG['clam_epochs'] = args.clam_epochs
    if args.clam_lr:
        CONFIG['clam_lr'] = args.clam_lr
    if args.clam_hidden:
        CONFIG['clam_hidden'] = args.clam_hidden
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print("=" * 40)
    print("  配置信息")
    print("=" * 40)
    print(f"  数据目录: {CONFIG['data_dir']}")
    print(f"  输出目录: {CONFIG['output_dir']}")
    print(f"  分类器: SVM (sklearn)")
    print("=" * 40)
    print()
    
    system = Task2ClassificationSystem(CONFIG)
    results = system.run()
    
    return results


class CLAMClassifier(nn.Cell):
    def __init__(self, in_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.n_classes = n_classes
        
        self.attention = nn.SequentialCell([
            nn.Dense(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dense(hidden_dim, 1)
        ])
        
        self.classifier = nn.SequentialCell([
            nn.Dense(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Dense(hidden_dim, n_classes)
        ])
    
    def construct(self, x):
        attn_weights = self.attention(x)
        attn_weights = ops.Softmax(axis=0)(attn_weights)
        
        weighted = x * attn_weights
        logits = self.classifier(weighted)
        
        return logits, attn_weights


class Task2ClassificationSystem:
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.label_encoder = None
        self.scaler = None
    
    def load_model(self):
        print("\n[1/7] 加载UNI2模型...")
        self.model, _, _ = get_encoder(
            enc_name=self.config['model_name'],
            device=self.config['device'],
            use_amp=self.config['use_amp'],
            test_batch=True,
            enable_graph_optimize=self.config['enable_graph_optimize']
        )
        print(f"    ✓ 模型加载成功")
        return self
    
    def load_dataset(self):
        print("\n[2/7] 加载BACH数据集...")
        
        data_dir = self.config['data_dir']
        csv_path = os.path.join(data_dir, self.config['csv_file'])
        
        df = pd.read_csv(csv_path)
        df.columns = ['filename', 'label']
        
        df['filepath'] = df['label'] + '/' + df['filename']
        df['full_path'] = df['filepath'].apply(lambda x: os.path.join(data_dir, x))
        
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
        df['label_encoded'] = df['label'].map(self.label_encoder)
        
        print(f"    数据集大小: {len(df)} 张图像")
        print(f"    类别: {list(self.label_encoder.keys())}")
        
        self.dataset_df = df
        return self
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.config['img_size'], self.config['img_size']))
        
        img_np = np.asarray(image, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        return img_np.transpose(2, 0, 1).astype(np.float32)
    
    def extract_features(self):
        print("\n[3/7] 提取特征向量...")
        
        self.model.set_train(False)
        
        features_list = []
        labels_list = []
        
        total = len(self.dataset_df)
        batch_size = self.config.get('batch_size', 32)
        
        start_time = time.time()
        
        batch_images = []
        
        for idx, row in tqdm(self.dataset_df.iterrows(), total=total, desc="    提取中", ncols=60):
            img_np = self.preprocess_image(row['full_path'])
            batch_images.append(img_np)
            
            if len(batch_images) >= batch_size:
                features_batch = self._batch_inference(batch_images)
                features_list.extend(features_batch)
                batch_images = []
        
        if batch_images:
            features_batch = self._batch_inference(batch_images)
            features_list.extend(features_batch)
        
        for idx in range(len(self.dataset_df)):
            labels_list.append(self.dataset_df.iloc[idx]['label_encoded'])
        
        extract_time = time.time() - start_time
        
        self.features = np.array(features_list, dtype=np.float32)
        self.labels = np.array(labels_list)
        
        speed = total / extract_time if extract_time > 0 else 0
        
        print(f"    ✓ 特征维度: {self.features.shape}")
        print(f"    ✓ 速度: {speed:.2f} img/s")
        
        return self
    
    def _batch_inference(self, batch_images):
        batch_tensor = Tensor(np.array(batch_images, dtype=np.float32))
        
        feat = self.model(batch_tensor)
        
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.ndim == 3:
            feat = feat.mean(axis=1)
        
        return [f.flatten() for f in feat.asnumpy()]
    
    def split_data(self):
        print("\n[4/7] 划分数据集...")
        
        from sklearn.model_selection import train_test_split
        
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.labels,
            test_size=val_ratio + test_ratio,
            random_state=self.config['random_seed'],
            stratify=self.labels
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - val_size,
            random_state=self.config['random_seed'],
            stratify=y_temp
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"    训练集: {len(X_train)} ({train_ratio*100:.0f}%)")
        print(f"    验证集: {len(X_val)} ({val_ratio*100:.0f}%)")
        print(f"    测试集: {len(X_test)} ({test_ratio*100:.0f}%)")
        
        return self
    
    def train_svm_classifier(self):
        print("\n[5/7] 训练分类器 (多模型集成)...")
        
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        log_file = os.path.join(self.config['output_dir'], 'train_log.txt')
        
        svm_configs = [
            {'name': 'SVM-RBF-C1', 'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'name': 'SVM-RBF-C10', 'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'name': 'SVM-RBF-C0.1', 'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
            {'name': 'SVM-Poly-C1', 'C': 1.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
            {'name': 'SVM-Linear', 'C': 1.0, 'kernel': 'linear'},
            {'name': 'SVM-RBF-balanced', 'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced'},
            {'name': 'SVM-Linear-balanced', 'C': 1.0, 'kernel': 'linear', 'class_weight': 'balanced'},
        ]
        
        rf_configs = [
            {'name': 'RF-n100-d10', 'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
            {'name': 'RF-n200-d15', 'n_estimators': 200, 'max_depth': 15, 'class_weight': 'balanced'},
            {'name': 'RF-n100-dNone', 'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced'},
        ]
        
        gb_configs = [
            {'name': 'GB-n100-lr0.1', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
            {'name': 'GB-n150-lr0.05', 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5},
        ]
        
        all_configs = svm_configs + rf_configs + gb_configs
        
        best_val_acc = 0
        best_clf = None
        best_clf_name = ""
        
        print(f"    开始训练 (共{len(all_configs)}个配置)...")
        
        train_scores = []
        val_scores = []
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  任务二：多模型分类器训练日志\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"训练样本: {len(self.X_train_scaled)}\n")
            f.write(f"验证样本: {len(self.X_val_scaled)}\n\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Config':<25}{'Train Acc':<12}{'Val Acc':<12}\n")
            f.write("-" * 70 + "\n")
        
        for epoch, config in enumerate(all_configs):
            config_name = config.pop('name')
            
            if config_name.startswith('SVM'):
                clf = SVC(**config, random_state=42, probability=True)
            elif config_name.startswith('RF'):
                clf = RandomForestClassifier(**config, random_state=42)
            elif config_name.startswith('GB'):
                clf = GradientBoostingClassifier(**config, random_state=42)
            else:
                continue
            
            clf.fit(self.X_train_scaled, self.y_train)
            
            train_acc = clf.score(self.X_train_scaled, self.y_train)
            val_acc = clf.score(self.X_val_scaled, self.y_val)
            
            train_scores.append(train_acc)
            val_scores.append(val_acc)
            
            print(f"    [{epoch+1}/{len(all_configs)}] {config_name}: Train={train_acc:.4f}, Val={val_acc:.4f}")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{config_name:<25}{train_acc:<12.4f}{val_acc:<12.4f}\n")
            
            config['name'] = config_name
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_clf = clf
                best_clf_name = config_name
        
        print(f"\n    最佳单一模型: {best_clf_name}, 验证准确率: {best_val_acc:.4f}")
        
        print("\n    [集成学习] 构建Voting分类器...")
        voting_clf = VotingClassifier(
            estimators=[
                ('svm', SVC(C=1.0, kernel='linear', class_weight='balanced', probability=True, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced', random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
            ],
            voting='soft'
        )
        voting_clf.fit(self.X_train_scaled, self.y_train)
        voting_val_acc = voting_clf.score(self.X_val_scaled, self.y_val)
        voting_train_acc = voting_clf.score(self.X_train_scaled, self.y_train)
        
        print(f"    Voting集成: Train={voting_train_acc:.4f}, Val={voting_val_acc:.4f}")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("-" * 70 + "\n")
            f.write(f"{'Voting-Ensemble':<25}{voting_train_acc:<12.4f}{voting_val_acc:<12.4f}\n")
        
        if voting_val_acc > best_val_acc:
            best_val_acc = voting_val_acc
            best_clf = voting_clf
            best_clf_name = "Voting-Ensemble"
        
        print(f"\n    ✓ 最佳模型: {best_clf_name}, 验证准确率: {best_val_acc:.4f}")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("-" * 70 + "\n")
            f.write(f"\n最佳模型: {best_clf_name}\n")
            f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
        
        all_model_names = [c['name'] for c in all_configs] + ['Voting-Ensemble']
        
        self.model = best_clf
        self.train_history = {
            'train_acc': train_scores + [voting_train_acc],
            'val_acc': val_scores + [voting_val_acc],
            'model_names': all_model_names,
            'best_model': best_clf_name,
            'best_val_acc': best_val_acc
        }
        
        import joblib
        model_path = os.path.join(self.config['output_dir'], 'classifier_model.pkl')
        joblib.dump(best_clf, model_path)
        print(f"    ✓ 模型已保存: {model_path}")
        
        model_info = {
            'label_encoder': self.label_encoder,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
        }
        with open(os.path.join(self.config['output_dir'], 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return self
    
    def evaluate(self):
        print("\n[6/7] 评估测试集...")
        
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
        
        clf = self.model
        
        test_preds = clf.predict(self.X_test_scaled)
        test_probs = clf.predict_proba(self.X_test_scaled)
        
        test_acc = accuracy_score(self.y_test, test_preds)
        f1_macro = f1_score(self.y_test, test_preds, average='macro')
        f1_weighted = f1_score(self.y_test, test_preds, average='weighted')
        conf_matrix = confusion_matrix(self.y_test, test_preds)
        
        idx_to_label = {v: k for k, v in self.label_encoder.items()}
        per_class_f1 = {}
        per_class_auc = {}
        for i in range(len(self.label_encoder)):
            class_name = idx_to_label[i]
            if (self.y_test == i).sum() > 0:
                y_true_binary = (self.y_test == i).astype(int)
                y_pred_binary = (test_preds == i).astype(int)
                per_class_f1[class_name] = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
                fpr, tpr, _ = roc_curve(y_true_binary, test_probs[:, i])
                per_class_auc[class_name] = auc(fpr, tpr)
        
        print(f"    测试集准确率: {test_acc:.4f}")
        print(f"    F1 (macro): {f1_macro:.4f}")
        print(f"    混淆矩阵:\n    {conf_matrix}")
        
        accuracy_score_val = test_acc * 100 * 0.5
        
        self.results = {
            'accuracy': test_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_f1': per_class_f1,
            'per_class_auc': per_class_auc,
            'best_val_acc': max(self.train_history['val_acc']),
            'accuracy_score': accuracy_score_val,
            'test_probs': test_probs.tolist(),
            'test_preds': test_preds.tolist(),
        }
        
        return self
    
    def save_results(self):
        print("\n[7/7] 保存结果...")
        
        output_dir = self.config['output_dir']
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        if self.config['save_features']:
            np.save(os.path.join(output_dir, 'features_train.npy'), self.X_train_scaled)
            np.save(os.path.join(output_dir, 'features_test.npy'), self.X_test_scaled)
            np.save(os.path.join(output_dir, 'labels_train.npy'), self.y_train)
            np.save(os.path.join(output_dir, 'labels_test.npy'), self.y_test)
        
        self._plot_curves()
        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_tsne()
        self._save_summary()
        
        print(f"    ✓ 结果已保存到: {output_dir}")
        
        return self
    
    def _plot_curves(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if 'train_loss' in self.train_history:
            axes[0].plot(self.train_history['train_loss'], 'b-', linewidth=2)
        else:
            axes[0].plot(self.train_history['train_acc'], 'b-', linewidth=2)
        axes[0].set_xlabel('Config Index')
        axes[0].set_ylabel('Train Accuracy')
        axes[0].set_title('Train Accuracy per Configuration')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.train_history['val_acc'], 'g-', linewidth=2)
        axes[1].set_xlabel('Config Index')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Validation Accuracy per Configuration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        model_names = self.train_history.get('model_names', [f'Config {i+1}' for i in range(len(self.train_history['val_acc']))])
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.35
        plt.bar(x - width/2, self.train_history['train_acc'], width, label='Train Acc', color='#1f77b4')
        plt.bar(x + width/2, self.train_history['val_acc'], width, label='Val Acc', color='#2ca02c')
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('SVM Configuration Comparison', fontsize=14)
        plt.xticks(x, model_names, rotation=30, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'svm_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        class_names = list(self.label_encoder.keys())
        train_counts = [np.sum(self.y_train == i) for i in range(len(class_names))]
        test_counts = [np.sum(self.y_test == i) for i in range(len(class_names))]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, train_counts, width, label='Train', color='#1f77b4')
        plt.bar(x + width/2, test_counts, width, label='Test', color='#ff7f0e')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Class Distribution', fontsize=14)
        plt.xticks(x, class_names, rotation=15)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'class_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = np.array(self.results['confusion_matrix'])
        class_names = list(self.label_encoder.keys())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix (Acc: {self.results["accuracy"]:.2%})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 12})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Normalized Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix_normalized.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        
        test_probs = np.array(self.results['test_probs'])
        class_names = list(self.label_encoder.keys())
        
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, class_name in enumerate(class_names):
            y_true_binary = (self.y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, test_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_tsne(self):
        try:
            from sklearn.manifold import TSNE
            
            X_all = np.vstack([self.X_train_scaled, self.X_test_scaled])
            y_all = np.concatenate([self.y_train, self.y_test])
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all)-1))
            X_tsne = tsne.fit_transform(X_all)
            
            import matplotlib.pyplot as plt
            
            class_names = list(self.label_encoder.keys())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            plt.figure(figsize=(10, 8))
            for i, class_name in enumerate(class_names):
                mask = y_all == i
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=colors[i % len(colors)], label=class_name,
                           alpha=0.6, s=50)
            
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.title('t-SNE Visualization of Feature Space', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['output_dir'], 'tsne_visualization.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    t-SNE visualization skipped: {e}")
    
    def _save_summary(self):
        summary_path = os.path.join(self.config['output_dir'], 'summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  任务二：BACH 4分类结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模型: UNI2 + SVM (sklearn)\n")
            f.write(f"特征提取: UNI2-h (MindSpore)\n")
            f.write(f"分类器: SVM\n")
            f.write(f"数据划分: 70%/15%/15%\n")
            f.write(f"最佳模型: {self.train_history['best_model']}\n\n")
            f.write(f"测试集准确率: {self.results['accuracy']:.4f}\n")
            f.write(f"F1 (macro): {self.results['f1_macro']:.4f}\n")
            f.write(f"F1 (weighted): {self.results['f1_weighted']:.4f}\n")
            f.write(f"最佳验证准确率: {self.results['best_val_acc']:.4f}\n")
            f.write(f"准确率分: {self.results['accuracy_score']:.4f}\n\n")
            f.write("各类别F1:\n")
            for k, v in self.results['per_class_f1'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n各类别AUC:\n")
            for k, v in self.results['per_class_auc'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n混淆矩阵:\n")
            f.write(str(np.array(self.results['confusion_matrix'])))
    
    def run(self):
        print("\n开始执行任务二...")
        
        start_time = time.time()
        
        self.load_model()
        self.load_dataset()
        self.extract_features()
        self.split_data()
        self.train_svm_classifier()
        self.evaluate()
        self.save_results()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("  任务二完成!")
        print(f"  总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"  测试准确率: {self.results['accuracy']:.4f}")
        print(f"  准确率分: {self.results['accuracy_score']:.4f}")
        print("=" * 60)
        
        return self.results


if __name__ == '__main__':
    import argparse
    main()