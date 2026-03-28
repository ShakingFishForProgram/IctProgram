import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GraphOptimizer:
    """整图下沉优化器 - 减少Python-NPU交互开销"""

    def __init__(self, model: nn.Module, device: str = 'npu:0'):
        self.model = model
        self.device = device
        self.compiled_model = None

    def optimize_for_inference(self, 
                              use_amp: bool = True,
                              enable_graph_compile: bool = True) -> nn.Module:
        """优化模型用于推理
        
        Args:
            use_amp: 是否启用自动混合精度
            enable_graph_compile: 是否启用图编译优化
            
        Returns:
            优化后的模型
        """
        self.model = self.model.to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model = self._apply_amp(self.model) if use_amp else self.model
        
        if enable_graph_compile:
            self.compiled_model = self._try_compile()
        else:
            self.compiled_model = self.model
            
        logger.info(f"Graph optimization completed on device: {self.device}")
        return self.compiled_model or self.model

    def _apply_amp(self, model: nn.Module) -> nn.Module:
        """应用自动混合精度"""
        if hasattr(torch, 'npu') and self.device.startswith('npu'):
            try:
                from torch.npu.amp import autocast, GradScaler
                model.use_amp = True
                model.amp_scaler = GradScaler()
                logger.info("AMP enabled for NPU")
            except ImportError:
                model.use_amp = True
                logger.info("AMP enabled using torch.cuda.amp (fallback)")
        elif torch.cuda.is_available():
            from torch.cuda.amp import autocast, GradScaler
            model.use_amp = True
            logger.info("AMP enabled for CUDA")
        else:
            model.use_amp = False
            logger.info("AMP not available, using FP32")
        return model

    def _try_compile(self) -> Optional[nn.Module]:
        """尝试使用torch.compile进行图编译"""
        try:
            if hasattr(torch, 'npu'):
                try:
                    compiled = torch.compile(self.model, backend='inductor')
                    logger.info("Model compiled with torch.compile (inductor backend)")
                    return compiled
                except Exception as e:
                    logger.warning(f"NPU torch.compile failed: {e}")
                    
            if torch.cuda.is_available():
                try:
                    compiled = torch.compile(self.model, backend='inductor')
                    logger.info("Model compiled with torch.compile (CUDA inductor)")
                    return compiled
                except Exception as e:
                    logger.warning(f"CUDA torch.compile failed: {e}")
                    
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")
            
        return None

    def inference_with_graph(self, x: torch.Tensor) -> torch.Tensor:
        """使用整图下沉进行推理
        
        Args:
            x: 输入张量
            
        Returns:
            模型输出
        """
        with torch.no_grad():
            if hasattr(self.model, 'use_amp') and self.model.use_amp:
                with torch.autocast(device_type='npu' if self.device.startswith('npu') else 'cuda', 
                                   dtype=torch.float16, 
                                   enabled=self.device.startswith('npu') or self.device.startswith('cuda')):
                    if self.compiled_model is not None:
                        output = self.compiled_model(x)
                    else:
                        output = self.model(x)
            else:
                if self.compiled_model is not None:
                    output = self.compiled_model(x)
                else:
                    output = self.model(x)
        return output

    def optimize_memory(self):
        """优化内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'npu') and self.device.startswith('npu'):
            try:
                torch.npu.empty_cache()
            except:
                pass
        logger.info("Memory cache cleared")

    def set_to_inference_mode(self):
        """设置模型为推理模式"""
        self.model.eval()
        
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                
        logger.info("Model set to inference mode")

    def get_optimization_info(self) -> Dict[str, Any]:
        """获取优化信息"""
        info = {
            'device': self.device,
            'compiled': self.compiled_model is not None,
            'amp_enabled': getattr(self.model, 'use_amp', False),
        }
        
        if torch.cuda.is_available():
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated() / (1024**2)
            info['cuda_memory_reserved'] = torch.cuda.memory_reserved() / (1024**2)
            
        return info
