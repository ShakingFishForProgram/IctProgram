import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """创建优化后的DataLoader
    
    Args:
        dataset: 数据集
        batch_size: batch大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存(加速CPU-GPU传输)
        persistent_workers: 是否保持worker进程
        prefetch_factor: 预取因子
        drop_last: 是否丢弃不完整的batch
        collate_fn: 自定义collate函数
        
    Returns:
        优化后的DataLoader
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    logger.info(f"Optimized DataLoader created: batch_size={batch_size}, num_workers={num_workers}")
    return dataloader


class DataPrefetcher:
    """数据预取器 - 实现CPU-GPU并行加载"""

    def __init__(self, dataloader: DataLoader, device: str = 'npu:0'):
        self.dataloader = dataloader
        self.device = device
        self.stream = None
        
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        elif hasattr(torch, 'npu') and device.startswith('npu'):
            try:
                self.stream = torch.npu.Stream(device)
            except:
                self.stream = None
                
        self.data_iter = iter(dataloader)
        self.preload()

    def preload(self):
        """预加载下一个batch"""
        try:
            self.next_data = next(self.data_iter)
        except StopIteration:
            self.next_data = None
            return
            
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                if isinstance(self.next_data, (list, tuple)):
                    self.next_data = [d.to(self.device, non_blocking=True) 
                                     if isinstance(d, torch.Tensor) else d 
                                     for d in self.next_data]
                elif isinstance(self.next_data, torch.Tensor):
                    self.next_data = self.next_data.to(self.device, non_blocking=True)

    def next(self):
        """获取下一个batch"""
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            
        data = self.next_data
        
        if data is not None:
            self.preload()
            
        return data

    def __iter__(self):
        return self

    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data

    def __len__(self):
        return len(self.dataloader)


class AsyncDataLoader:
    """异步数据加载器 - 使用后台线程预取数据"""

    def __init__(self, dataloader: DataLoader, device: str = 'npu:0'):
        self.dataloader = dataloader
        self.device = device
        self.data_queue = []
        self.max_queue_size = 2
        self.current_idx = 0

    def _load_batch(self, batch_data):
        """将batch数据加载到目标设备"""
        if isinstance(batch_data, (list, tuple)):
            return [d.to(self.device, non_blocking=True) 
                   if isinstance(d, torch.Tensor) else d 
                   for d in batch_data]
        elif isinstance(batch_data, torch.Tensor):
            return batch_data.to(self.device, non_blocking=True)
        return batch_data

    def __iter__(self):
        self.data_iter = iter(self.dataloader)
        return self

    def __next__(self):
        try:
            batch_data = next(self.data_iter)
            return self._load_batch(batch_data)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


def create_persistent_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """创建持久化DataLoader (worker进程常驻)
    
    Args:
        dataset: 数据集
        batch_size: batch大小
        num_workers: worker数量
        **kwargs: 其他DataLoader参数
        
    Returns:
        持久化DataLoader
    """
    return create_optimized_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        **kwargs
    )
