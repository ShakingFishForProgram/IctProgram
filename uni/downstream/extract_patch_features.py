import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

from ..get_encoder import get_encoder

torch.multiprocessing.set_sharing_strategy("file_system")


@torch.no_grad()
def extract_patch_features_from_dataloader(
    model, 
    dataloader,
    use_amp: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """
    Uses model to extract features+labels from images iterated over the dataloader.
    Optimized version with pre-allocated arrays, AMP support, and DataLoader optimization.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.
        use_amp (bool): Whether to enable automatic mixed precision for faster inference.
        pin_memory (bool): Whether to use pinned memory for faster CPU-GPU transfer.
        prefetch_factor (int): Number of batches to prefetch per worker.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    device = next(model.parameters())[0].device
    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    batch_size = dataloader.batch_size
    
    total_samples = len(dataloader.dataset)
    
    if hasattr(model, 'num_features'):
        feat_dim = model.num_features
    elif hasattr(model, 'embed_dim'):
        feat_dim = model.embed_dim
    else:
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat_dim = model(dummy).shape[-1]
    
    all_embeddings = np.zeros((total_samples, feat_dim), dtype=np.float32)
    all_labels = np.zeros(total_samples, dtype=np.int64)
    
    idx = 0
    
    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Extracting features"
    ):
        batch = batch.to(device, non_blocking=True)
        
        if use_amp:
            if device_type == 'npu':
                with torch.npu.amp.autocast():
                    embeddings = model(batch)
            elif device_type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    embeddings = model(batch)
            else:
                embeddings = model(batch)
        else:
            embeddings = model(batch)
        
        remaining = batch.shape[0]
        
        all_embeddings[idx:idx + remaining] = embeddings.cpu().numpy()
        all_labels[idx:idx + remaining] = target.numpy()[:remaining]
        
        idx += remaining
        
        assert not np.isnan(all_embeddings[idx - remaining:idx]).any()

    asset_dict = {
        "embeddings": all_embeddings,
        "labels": all_labels,
    }

    return asset_dict


def extract_patch_features_optimized(
    model,
    dataloader,
    use_amp: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Complete optimized feature extraction function with DataLoader optimization.

    Args:
        model: Pre-trained model
        dataloader: DataLoader
        use_amp: Enable mixed precision inference
        num_workers: Number of DataLoader workers
        pin_memory: Use pinned memory

    Returns:
        dict: {'embeddings': ..., 'labels': ...}
    """
    original_num_workers = dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
    original_pin_memory = dataloader.pin_memory if hasattr(dataloader, 'pin_memory') else False
    
    if hasattr(dataloader, 'num_workers'):
        dataloader.num_workers = min(num_workers, 8)
    
    if hasattr(dataloader, 'pin_memory'):
        dataloader.pin_memory = pin_memory and (torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available()))
    
    results = extract_patch_features_from_dataloader(
        model, 
        dataloader, 
        use_amp=use_amp,
        pin_memory=pin_memory,
    )
    
    if hasattr(dataloader, 'num_workers'):
        dataloader.num_workers = original_num_workers
    if hasattr(dataloader, 'pin_memory'):
        dataloader.pin_memory = original_pin_memory
    
    return results


def create_optimized_dataloader(dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=False):
    """
    Create an optimized DataLoader for feature extraction.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Use pinned memory for faster transfer
        shuffle: Whether to shuffle data

    Returns:
        DataLoader: Optimized DataLoader
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and (torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available())),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
