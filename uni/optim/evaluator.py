import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'npu:0',
    num_classes: int = 4,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """评估模型在数据集上的性能
    
    Args:
        model: 待评估模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数
        class_names: 类别名称列表
        
    Returns:
        评估指标字典
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
        
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-Score (macro): {f1_macro:.4f}")
    logger.info(f"F1-Score (weighted): {f1_weighted:.4f}")
    
    return results


def benchmark(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'npu:0',
    num_runs: int = 10,
    warmup_runs: int = 5,
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """性能基准测试
    
    Args:
        model: 待测试模型
        dataloader: 数据加载器
        device: 设备
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数
        batch_size: 指定batch大小
        
    Returns:
        性能指标字典
    """
    model.eval()
    
    if batch_size is None:
        batch_size = dataloader.batch_size
    
    times = []
    throughputs = []
    
    if batch_size is not None:
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    else:
        for i, (images, _) in enumerate(dataloader):
            if i >= warmup_runs:
                break
            images = images.to(device)
            _ = model(images)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, 'npu'):
        try:
            torch.npu.synchronize()
        except:
            pass
            
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_runs:
                break
                
            images = images.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                outputs = model(images)
                end.record()
                
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                
            elif hasattr(torch, 'npu') and device.startswith('npu'):
                try:
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True, device=device)
                    end = torch.npu.Event(enable_timing=True, device=device)
                    
                    start.record()
                    outputs = model(images)
                    end.record()
                    
                    torch.npu.synchronize()
                    elapsed = start.elapsed_time(end)
                except:
                    s = time.time()
                    outputs = model(images)
                    elapsed = (time.time() - s) * 1000
            else:
                s = time.time()
                outputs = model(images)
                elapsed = (time.time() - s) * 1000
            
            times.append(elapsed)
            if elapsed > 0:
                throughputs.append((images.size(0) * 1000) / elapsed)
    
    avg_time = sum(times) / len(times) if times else 0
    throughput = sum(throughputs) / len(throughputs) if throughputs else 0
    
    results = {
        'num_runs': len(times),
        'avg_time_ms': avg_time,
        'throughput_imgs_per_sec': throughput,
        'min_time_ms': min(times) if times else 0,
        'max_time_ms': max(times) if times else 0,
        'std_time_ms': np.std(times) if len(times) > 1 else 0,
    }
    
    logger.info(f"Average inference time: {avg_time:.2f} ms")
    logger.info(f"Throughput: {throughput:.1f} img/s")
    
    return results


class PerformanceProfiler:
    """性能分析器 - 分析各层耗时"""

    def __init__(self, model: nn.Module, device: str = 'npu:0'):
        self.model = model
        self.device = device
        self.layer_times = {}
        self.hooks = []

    def _create_hook(self, name: str):
        def hook(module, input, output):
            if hasattr(self, '_start_time'):
                elapsed = time.time() - self._start_time
                if name not in self.layer_times:
                    self.layer_times[name] = []
                self.layer_times[name].append(elapsed)
        return hook

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """注册hooks到指定层"""
        if layer_names is None:
            for name, module in self.model.named_modules():
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
        else:
            for name, module in self.model.named_modules():
                if any(ln in name for ln in layer_names):
                    hook = module.register_forward_hook(self._create_hook(name))
                    self.hooks.append(hook)

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def profile(self, input_data: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
        """分析模型性能
        
        Args:
            input_data: 输入数据
            num_runs: 运行次数
            
        Returns:
            各层平均耗时字典
        """
        self.layer_times = {}
        self.register_hooks()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_runs):
                self._start_time = time.time()
                _ = self.model(input_data.to(self.device))
                
        self.remove_hooks()
        
        avg_times = {}
        for name, times in self.layer_times.items():
            avg_times[name] = sum(times) / len(times) * 1000
            
        sorted_times = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_times[:20])


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: str = 'npu:0',
    num_runs: int = 10
) -> Dict[str, Dict[str, Any]]:
    """对比多个模型的性能
    
    Args:
        models: 模型名称到模型的字典
        dataloader: 数据加载器
        device: 设备
        num_runs: 测试次数
        
    Returns:
        各模型的性能对比结果
    """
    results = {}
    
    for name, model in models.items():
        logger.info(f"Benchmarking {name}...")
        model = model.to(device)
        results[name] = benchmark(model, dataloader, device, num_runs)
        
    return results
