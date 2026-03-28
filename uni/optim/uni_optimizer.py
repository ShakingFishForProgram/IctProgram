import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

from .quantizer import PTQQuantizer
from .pruner import StructuredPruner
from .graph_optimizer import GraphOptimizer
from .batch_inference import BatchOptimizer
from .data_prefetcher import create_optimized_dataloader
from .evaluator import evaluate_model, benchmark

logger = logging.getLogger(__name__)


class UNIOptimizer:
    """UNI模型推理加速优化器 - 整合所有优化技术"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'npu:0',
                 enc_name: str = 'uni2-h'):
        self.device = torch.device(device)
        self.enc_name = enc_name
        self.model_path = model_path
        self.model = None
        self.transform = None
        
        self.quantizer = None
        self.pruner = None
        self.graph_optimizer = None
        self.batch_optimizer = None
        
        self.optimization_applied = {
            'quantization': False,
            'pruning': False,
            'graph_optimization': False,
            'batch_optimization': False
        }
        
    def load_model(self, 
                  checkpoint: str = 'pytorch_model.bin',
                  assets_dir: Optional[str] = None,
                  use_amp: bool = True,
                  test_batch: bool = True) -> nn.Module:
        """加载UNI模型
        
        Args:
            checkpoint: 检查点文件名
            assets_dir: 检查点目录
            use_amp: 是否使用混合精度
            test_batch: 是否测试batch
            
        Returns:
            加载的模型
        """
        from uni import get_encoder
        
        self.model, self.transform = get_encoder(
            enc_name=self.enc_name,
            checkpoint=checkpoint,
            device=self.device,
            use_amp=use_amp,
            test_batch=test_batch,
            assets_dir=assets_dir
        )
        
        logger.info(f"Model {self.enc_name} loaded successfully")
        return self.model
    
    def apply_quantization(self, 
                          calibration_data: List[torch.Tensor],
                          weight_quant: str = 'per_channel_symmetric',
                          act_quant: str = 'per_tensor_affine',
                          num_calibration_batches: int = 10,
                          mix_precision: bool = True) -> nn.Module:
        """应用PTQ量化
        
        Args:
            calibration_data: 校准数据
            weight_quant: 权重量化方式
            act_quant: 激活量化方式
            num_calibration_batches: 校准batch数
            mix_precision: 是否使用混合精度
            
        Returns:
            量化后的模型
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.quantizer = PTQQuantizer(self.model, self.device)
        
        if mix_precision:
            sensitive_layers = ['layernorm', 'softmax', 'norm']
            self.model = self.quantizer.quantize_partial(
                calibration_data,
                sensitive_layers,
                num_calibration_batches
            )
        else:
            self.model = self.quantizer.quantize(
                calibration_data,
                weight_quant,
                act_quant,
                num_calibration_batches
            )
            
        self.optimization_applied['quantization'] = True
        logger.info("PTQ quantization applied")
        return self.model
    
    def apply_structured_pruning(self, 
                                dataloader: torch.utils.data.DataLoader,
                                pruning_ratio: float = 0.2,
                                prune_heads: bool = True,
                                prune_ffn: bool = False,
                                finetune_epochs: int = 0) -> nn.Module:
        """应用结构化剪枝
        
        Args:
            dataloader: 数据加载器
            pruning_ratio: 剪枝比例
            prune_heads: 是否剪枝注意力头
            prune_ffn: 是否剪枝FFN
            finetune_epochs: 剪枝后微调轮数
            
        Returns:
            剪枝后的模型
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.pruner = StructuredPruner(self.model, pruning_ratio, self.device)
        
        if prune_heads:
            importance = self.pruner.compute_head_importance(dataloader)
            self.pruner.prune_heads(importance, pruning_ratio)
            
        if prune_ffn:
            self.pruner.prune_ffn(pruning_ratio=pruning_ratio * 0.5)
            
        if finetune_epochs > 0:
            self.pruner.finetune_after_pruning(dataloader, epochs=finetune_epochs)
            
        self.optimization_applied['pruning'] = True
        logger.info(f"Structured pruning applied (ratio: {pruning_ratio})")
        return self.model
    
    def apply_graph_optimization(self, 
                                use_amp: bool = True,
                                enable_compile: bool = True) -> nn.Module:
        """应用整图下沉优化
        
        Args:
            use_amp: 是否使用混合精度
            enable_compile: 是否启用图编译
            
        Returns:
            优化后的模型
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.graph_optimizer = GraphOptimizer(self.model, self.device)
        self.model = self.graph_optimizer.optimize_for_inference(
            use_amp=use_amp,
            enable_graph_compile=enable_compile
        )
        
        self.optimization_applied['graph_optimization'] = True
        logger.info("Graph optimization applied")
        return self.model
    
    def apply_batch_optimization(self, 
                                 dataloader: torch.utils.data.DataLoader,
                                 max_batch: int = 64) -> int:
        """应用Batch推理优化
        
        Args:
            dataloader: 数据加载器
            max_batch: 最大测试batch大小
            
        Returns:
            最优batch size
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.batch_optimizer = BatchOptimizer(self.model, self.device)
        optimal_bs, _ = self.batch_optimizer.find_optimal_batch_size(
            max_batch=max_batch
        )
        
        self.optimization_applied['batch_optimization'] = True
        logger.info(f"Batch optimization applied (optimal batch size: {optimal_bs})")
        return optimal_bs
    
    def optimize_all(self, 
                    dataloader: Optional[torch.utils.data.DataLoader] = None,
                    calibration_data: Optional[List[torch.Tensor]] = None,
                    quantization: bool = True,
                    pruning: bool = False,
                    graph_optimize: bool = True,
                    batch_optimize: bool = True,
                    pruning_ratio: float = 0.2,
                    batch_size: int = 32) -> nn.Module:
        """应用所有优化技术
        
        Args:
            dataloader: 数据加载器
            calibration_data: 校准数据
            quantization: 是否应用量化
            pruning: 是否应用剪枝
            graph_optimize: 是否应用图优化
            batch_optimize: 是否应用Batch优化
            pruning_ratio: 剪枝比例
            batch_size: 默认batch大小
            
        Returns:
            优化后的模型
        """
        if graph_optimize:
            self.apply_graph_optimization()
            
        if quantization:
            if calibration_data is None and dataloader is not None:
                calibration_data = []
                for i, (images, _) in enumerate(dataloader):
                    if i >= 10:
                        break
                    calibration_data.append(images)
            self.apply_quantization(calibration_data or [])
            
        if pruning and dataloader is not None:
            self.apply_structured_pruning(dataloader, pruning_ratio)
            
        if batch_optimize and dataloader is not None:
            self.apply_batch_optimization(dataloader)
            
        return self.model
    
    def inference(self, images: torch.Tensor) -> torch.Tensor:
        """推理
        
        Args:
            images: 输入图像
            
        Returns:
            模型输出
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            
            if self.graph_optimizer is not None:
                output = self.graph_optimizer.inference_with_graph(images)
            else:
                output = self.model(images)
                
        return output
    
    def evaluate(self, 
                dataloader: torch.utils.data.DataLoader,
                num_classes: int = 4,
                class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            dataloader: 数据加载器
            num_classes: 类别数
            class_names: 类别名称
            
        Returns:
            评估指标
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        return evaluate_model(self.model, dataloader, self.device, num_classes, class_names)
    
    def benchmark(self, 
                 dataloader: torch.utils.data.DataLoader,
                 num_runs: int = 10,
                 warmup_runs: int = 5) -> Dict[str, Any]:
        """性能基准测试
        
        Args:
            dataloader: 数据加载器
            num_runs: 测试次数
            warmup_runs: 预热次数
            
        Returns:
            性能指标
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        return benchmark(self.model, dataloader, self.device, num_runs, warmup_runs)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化配置摘要"""
        summary = {
            'device': str(self.device),
            'model': self.enc_name,
            'optimizations_applied': self.optimization_applied,
            'quantization': {
                'enabled': self.optimization_applied['quantization'],
                'model_size_info': self.quantizer.get_model_size() if self.quantizer else None
            },
            'pruning': {
                'enabled': self.optimization_applied['pruning'],
                'info': self.pruner.get_pruning_info() if self.pruner else None
            },
            'graph_optimization': {
                'enabled': self.optimization_applied['graph_optimization'],
                'info': self.graph_optimizer.get_optimization_info() if self.graph_optimizer else None
            },
            'batch_optimization': {
                'enabled': self.optimization_applied['batch_optimization'],
                'optimal_batch': self.batch_optimizer.optimal_batch_size if self.batch_optimizer else None
            }
        }
        
        return summary


def create_optimizer(device: str = 'npu:0', enc_name: str = 'uni2-h') -> UNIOptimizer:
    """创建UNI优化器的便捷函数
    
    Args:
        device: 设备
        enc_name: 模型名称
        
    Returns:
        UNIOptimizer实例
    """
    return UNIOptimizer(device=device, enc_name=enc_name)
