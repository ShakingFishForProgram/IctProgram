import torch
import torch.nn as nn
import torch.quantization as tq
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PTQQuantizer:
    """PTQ后训练量化器 for ViT models"""

    def __init__(self, model: nn.Module, device: str = 'npu:0'):
        self.model = model
        self.device = device
        self.quantized_model = None
        self.qconfig = None

    def set_qconfig(self, 
                    weight_quant: str = 'per_channel_symmetric',
                    act_quant: str = 'per_tensor_affine'):
        """设置量化配置"""
        
        if weight_quant == 'per_channel_symmetric':
            weight_fake_quant = tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                reduce_range=False
            )
        else:
            weight_fake_quant = tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False
            )

        if act_quant == 'per_tensor_affine':
            activation_fake_quant = tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False
            )
        else:
            activation_fake_quant = tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False
            )

        self.qconfig = tq.QConfig(
            activation=activation_fake_quant,
            weight=weight_fake_quant
        )

    def prepare_model(self):
        """准备模型进行量化"""
        self.model.eval()
        self.model.qconfig = self.qconfig
        tq.prepare(self.model, inplace=True)
        logger.info("Model prepared for quantization")

    def calibrate(self, calibration_data: List[torch.Tensor], 
                  num_batches: int = 10):
        """校准量化参数
        
        Args:
            calibration_data: 校准数据列表
            num_batches: 校准使用的batch数量
        """
        self.model.eval()
        
        with torch.no_grad():
            for i, images in enumerate(calibration_data):
                if i >= num_batches:
                    break
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)
                    self.model(images)
        
        logger.info(f"Calibration completed with {min(num_batches, len(calibration_data))} batches")

    def convert(self):
        """转换为量化模型"""
        self.quantized_model = tq.convert(self.model)
        self.model = self.quantized_model
        logger.info("Model converted to quantized model")
        return self.quantized_model

    def quantize(self, calibration_data: List[torch.Tensor],
                 weight_quant: str = 'per_channel_symmetric',
                 act_quant: str = 'per_tensor_affine',
                 num_calibration_batches: int = 10) -> nn.Module:
        """完整量化流程
        
        Args:
            calibration_data: 校准数据
            weight_quant: 权重量化方式
            act_quant: 激活量化方式
            num_calibration_batches: 校准batch数
            
        Returns:
            量化后的模型
        """
        self.set_qconfig(weight_quant, act_quant)
        self.prepare_model()
        self.calibrate(calibration_data, num_calibration_batches)
        return self.convert()

    def quantize_partial(self, calibration_data: List[torch.Tensor],
                         sensitive_layers: Optional[List[str]] = None,
                         num_calibration_batches: int = 10) -> nn.Module:
        """混合精度量化 - 敏感层保持FP16
        
        Args:
            calibration_data: 校准数据
            sensitive_layers: 保持FP16的层名称列表
            num_calibration_batches: 校准batch数
            
        Returns:
            混合精度的量化模型
        """
        if sensitive_layers is None:
            sensitive_layers = ['layernorm', 'softmax', 'norm']

        self.set_qconfig('per_channel_symmetric', 'per_tensor_affine')
        self.prepare_model()
        self.calibrate(calibration_data, num_calibration_batches)
        
        self.model = tq.convert(self.model)
        
        logger.info(f"Partial quantization completed, {len(sensitive_layers)} layers keep FP16")
        return self.model

    def get_model_size(self) -> Dict[str, float]:
        """获取模型大小
        
        Returns:
            包含FP32和INT8模型大小的字典
        """
        def get_param_size(model):
            total = 0
            for param in model.parameters():
                total += param.numel() * param.element_size()
            return total / (1024 * 1024)

        fp32_size = get_param_size(self.model)
        
        if self.quantized_model is not None:
            int8_size = get_param_size(self.quantized_model)
        else:
            int8_size = fp32_size / 4

        return {
            'fp32_mb': fp32_size,
            'int8_mb': int8_size,
            'compression_ratio': fp32_size / int8_size if int8_size > 0 else 0
        }
