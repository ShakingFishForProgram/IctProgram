import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """Batch推理优化器 - 寻找最优batch size并执行批量推理"""

    def __init__(self, model: nn.Module, device: str = 'npu:0'):
        self.model = model
        self.device = device
        self.optimal_batch_size = None
        self.benchmark_results = []

    def find_optimal_batch_size(self, 
                                max_batch: int = 64,
                                input_shape: Tuple[int, int, int, int] = (3, 224, 224),
                                warmup_runs: int = 5,
                                test_runs: int = 10) -> Tuple[int, List[Tuple[int, float]]]:
        """寻找最优batch size
        
        Args:
            max_batch: 最大测试的batch size
            input_shape: 输入图像形状 (C, H, W)
            warmup_runs: 预热运行次数
            test_runs: 测试运行次数
            
        Returns:
            (最优batch size, 所有测试结果列表)
        """
        results = []
        batch_sizes = [1, 2, 4, 8, 16, 32, 64][:max_batch]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
            
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif hasattr(torch, 'npu'):
                    try:
                        torch.npu.synchronize()
                    except:
                        pass
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(test_runs):
                    _ = self.model(dummy_input)
                end.record()
                
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / test_runs
                
            elif hasattr(torch, 'npu') and self.device.startswith('npu'):
                try:
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True, device=self.device)
                    end = torch.npu.Event(enable_timing=True, device=self.device)
                    
                    start.record()
                    for _ in range(test_runs):
                        _ = self.model(dummy_input)
                    end.record()
                    
                    torch.npu.synchronize()
                    elapsed = start.elapsed_time(end) / test_runs
                except:
                    import time
                    start = time.time()
                    for _ in range(test_runs):
                        _ = self.model(dummy_input)
                    elapsed = (time.time() - start) * 1000 / test_runs
            else:
                import time
                start = time.time()
                for _ in range(test_runs):
                    _ = self.model(dummy_input)
                elapsed = (time.time() - start) * 1000 / test_runs
            
            throughput = (batch_size * 1000) / elapsed if elapsed > 0 else 0
            results.append((batch_size, throughput, elapsed))
            
            logger.info(f"Batch size: {batch_size}, Throughput: {throughput:.1f} img/s, Time: {elapsed:.2f} ms")
            
            del dummy_input
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.benchmark_results = results
        optimal = max(results, key=lambda x: x[1])
        self.optimal_batch_size = optimal[0]
        
        logger.info(f"Optimal batch size: {self.optimal_batch_size}")
        return self.optimal_batch_size, results

    def batch_inference(self, 
                       dataloader: torch.utils.data.DataLoader,
                       batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量推理
        
        Args:
            dataloader: 数据加载器
            batch_size: 使用的batch size，若为None则使用最优值
            
        Returns:
            (所有输出拼接, 所有标签拼接)
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size or 1
            
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                if images.size(0) != batch_size:
                    continue
                    
                images = images.to(self.device)
                outputs = self.model(images)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels)
                
        return torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)

    def inference_single(self, x: torch.Tensor) -> torch.Tensor:
        """单张推理
        
        Args:
            x: 输入图像 (C, H, W) 或 (B, C, H, W)
            
        Returns:
            模型输出
        """
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            
        return output

    def inference_batch(self, x: torch.Tensor) -> torch.Tensor:
        """批量推理 (输入单个batch)
        
        Args:
            x: 输入图像批次
            
        Returns:
            模型输出
        """
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            
        return output

    def get_throughput(self, 
                      batch_size: int = 32,
                      input_shape: Tuple[int, int, int, int] = (3, 224, 224),
                      num_runs: int = 100) -> Dict[str, float]:
        """计算指定batch size的吞吐量
        
        Args:
            batch_size: batch大小
            input_shape: 输入形状
            num_runs: 运行次数
            
        Returns:
            吞吐量统计信息
        """
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        for _ in range(10):
            _ = self.model(dummy_input)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        import time
        start = time.time()
        
        for _ in range(num_runs):
            _ = self.model(dummy_input)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = time.time() - start
        total_images = num_runs * batch_size
        throughput = total_images / elapsed
        
        return {
            'batch_size': batch_size,
            'num_runs': num_runs,
            'total_time_sec': elapsed,
            'images_per_second': throughput,
            'ms_per_batch': (elapsed / num_runs) * 1000,
            'ms_per_image': (elapsed / total_images) * 1000
        }

    def get_optimal_batch_info(self) -> Optional[Dict[str, Any]]:
        """获取最优batch size的信息"""
        if self.optimal_batch_size is None:
            return None
            
        for bs, tp, time_ms in self.benchmark_results:
            if bs == self.optimal_batch_size:
                return {
                    'batch_size': bs,
                    'throughput': tp,
                    'time_per_batch_ms': time_ms
                }
        return None
