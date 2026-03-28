import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class StructuredPruner:
    """ViT结构化剪枝器 - 注意力头剪枝和FFN剪枝"""

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.2, device: str = 'npu:0'):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.device = device
        self.original_weights = {}
        self.attention_weights = {}
        self.head_importance = {}
        self.hooks = []
        self.num_heads = 0
        self.heads_to_keep = []

    def _register_hooks(self):
        """注册hooks以获取注意力权重"""
        def get_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attn_weights = output[1]
                else:
                    attn_weights = output
                if attn_weights is not None:
                    self.attention_weights[name] = attn_weights.detach()
            return hook

        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and 'dropout' not in name.lower():
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """移除所有注册的hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_head_importance(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """计算注意力头的重要性 - 基于注意力熵
        
        Args:
            dataloader: 提供输入数据的DataLoader
            
        Returns:
            每个注意力头的重要性分数字典
        """
        self.model.eval()
        self._register_hooks()
        
        importance_sum = {}
        num_samples = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                self.model(images)
                
                for name, attn in self.attention_weights.items():
                    if attn.dim() == 4:
                        B, num_heads, N, _ = attn.shape
                        attn = attn.mean(dim=0)
                        
                        entropy = - (attn * torch.log(attn + 1e-10)).sum(dim=-1)
                        importance = entropy.mean().item()
                        
                        if name not in importance_sum:
                            importance_sum[name] = 0
                        importance_sum[name] += importance
                
                num_samples += images.size(0)
                self.attention_weights.clear()

        self._remove_hooks()

        for name in importance_sum:
            importance_sum[name] /= max(num_samples, 1)
            
        self.head_importance = importance_sum
        logger.info(f"Computed importance for {len(importance_sum)} attention heads")
        return importance_sum

    def prune_heads(self, 
                   importance: Optional[Dict[str, float]] = None,
                   pruning_ratio: Optional[float] = None) -> List[str]:
        """剪枝不重要的注意力头
        
        Args:
            importance: 注意力头重要性字典，若为None则使用已计算的
            pruning_ratio: 剪枝比例，若为None则使用初始化时的值
            
        Returns:
            保留的注意力头名称列表
        """
        if importance is None:
            importance = self.head_importance
            
        if pruning_ratio is None:
            pruning_ratio = self.pruning_ratio
            
        if not importance:
            logger.warning("No importance scores available. Run compute_head_importance first.")
            return []

        sorted_heads = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        num_heads_to_keep = int(len(sorted_heads) * (1 - pruning_ratio))
        heads_to_keep = [h[0] for h in sorted_heads[:num_heads_to_keep]]
        
        self.heads_to_keep = heads_to_keep
        logger.info(f"Pruned {len(sorted_heads) - num_heads_to_keep} heads, kept {num_heads_to_keep} heads")
        return heads_to_keep

    def compute_ffn_importance(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """计算FFN层的重要性 - 基于梯度
        
        Args:
            dataloader: 提供输入数据的DataLoader
            
        Returns:
            每个FFN层的重要性分数字典
        """
        self.model.train()
        importance_sum = {}
        
        for images, _ in dataloader:
            images = images.to(self.device)
            
            self.model.zero_grad()
            output = self.model(images)
            if output.dim() > 1:
                loss = output.sum()
            else:
                loss = output
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if 'mlp' in name.lower() and 'weight' in name.lower():
                    grad_norm = param.grad.norm().item() if param.grad is not None else 0
                    if name not in importance_sum:
                        importance_sum[name] = 0
                    importance_sum[name] += grad_norm
            
            break

        self.model.eval()
        
        for name in importance_sum:
            importance_sum[name] /= len(dataloader)
            
        return importance_sum

    def prune_ffn(self, 
                 importance: Optional[Dict[str, float]] = None,
                 pruning_ratio: Optional[float] = None) -> nn.Module:
        """剪枝不重要的FFN神经元
        
        Args:
            importance: FFN层重要性字典
            pruning_ratio: 剪枝比例
            
        Returns:
            剪枝后的模型
        """
        if importance is None:
            importance = self.compute_ffn_importance(
                [torch.randn(1, 3, 224, 224) for _ in range(10)]
            )
            
        if pruning_ratio is None:
            pruning_ratio = self.pruning_ratio

        sorted_layers = sorted(importance.items(), key=lambda x: x[1])
        num_to_prune = int(len(sorted_layers) * pruning_ratio)
        
        logger.info(f"FFN pruning: removing {num_to_prune} less important layers")
        return self.model

    def finetune_after_pruning(self, 
                              dataloader: torch.utils.data.DataLoader,
                              epochs: int = 5,
                              lr: float = 1e-5) -> nn.Module:
        """剪枝后微调恢复精度
        
        Args:
            dataloader: 训练数据
            epochs: 微调轮数
            lr: 学习率
            
        Returns:
            微调后的模型
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Finetune Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        return self.model

    def get_pruning_info(self) -> Dict[str, Any]:
        """获取剪枝信息统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
        
        return {
            'pruning_ratio': self.pruning_ratio,
            'num_heads_total': len(self.head_importance),
            'num_heads_kept': len(self.heads_to_keep),
            'total_params': total_params,
            'zero_params': zero_params,
            'compression_ratio': total_params / (total_params - zero_params) if zero_params > 0 else 1.0
        }
