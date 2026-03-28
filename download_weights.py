#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载并转换UNI模型权重
====================
从HuggingFace下载PyTorch模型，并转换为MindSpore格式

使用：
    python download_weights.py
"""

import os
import sys

def download_and_convert():
    """下载并转换模型"""
    
    # ==================== 第1步：下载PyTorch模型 ====================
    print("\n" + "=" * 50)
    print("  第1步：下载PyTorch模型")
    print("=" * 50)
    
    try:
        from huggingface_hub import login, hf_hub_download
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface_hub")
        return
    
    token = input("请输入HuggingFace Token: ").strip()
    if not token:
        print("Token不能为空！")
        return
    
    print("正在登录HuggingFace...")
    try:
        login(token=token)
    except Exception as e:
        print(f"登录失败: {e}")
        return
    
    assets_dir = "./assets/ckpts"
    ckpt_dir = os.path.join(assets_dir, "vit_large_patch16_224.dinov2.uni_mass100k")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    pytorch_model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    
    # 尝试下载UNI
    print("\n正在下载UNI模型...")
    try:
        hf_hub_download(
            repo_id='MahmoodLab/UNI',
            filename='pytorch_model.bin',
            local_dir=ckpt_dir,
            force_download=True
        )
        model_name = "UNI"
        print(f"\n✓ UNI模型下载完成！")
    except Exception as e:
        print(f"UNI下载失败: {e}")
        
        # 尝试下载UNI2-h
        print("\n正在下载UNI2-h模型...")
        try:
            hf_hub_download(
                repo_id='MahmoodLab/UNI2-h',
                filename='pytorch_model.bin',
                local_dir=ckpt_dir,
                force_download=True
            )
            model_name = "UNI2-h"
            print(f"\n✓ UNI2-h模型下载完成！")
        except Exception as e2:
            print(f"UNI2-h下载也失败: {e2}")
            return
    
    pytorch_model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    
    if not os.path.exists(pytorch_model_path):
        print(f"错误: 模型文件不存在 {pytorch_model_path}")
        return
    
    print(f"模型路径: {pytorch_model_path}")
    file_size = os.path.getsize(pytorch_model_path) / (1024*1024*1024)
    print(f"模型大小: {file_size:.2f} GB")
    
    # ==================== 第2步：转换为MindSpore格式 ====================
    print("\n" + "=" * 50)
    print("  第2步：转换为MindSpore格式")
    print("=" * 50)
    
    convert_pytorch_to_mindspore(pytorch_model_path, ckpt_dir, model_name)


def convert_pytorch_to_mindspore(pytorch_path, output_dir, model_name):
    """将PyTorch模型转换为MindSpore格式"""
    
    print("\n开始转换模型...")
    
    try:
        import torch
    except ImportError:
        print("请安装PyTorch: pip install torch")
        return
    
    try:
        import mindspore as ms
        from mindspore import Tensor, nn
    except ImportError:
        print("请安装MindSpore: pip install mindspore")
        return
    
    # 加载PyTorch模型
    print("    加载PyTorch模型...")
    try:
        state_dict = torch.load(pytorch_path, map_location='cpu')
        print(f"    ✓ 加载成功，参数数量: {len(state_dict)}")
    except Exception as e:
        print(f"    ✗ 加载失败: {e}")
        return
    
    # 创建MindSpore参数映射
    print("    创建参数映射...")
    
    ms_params = {}
    
    for key, value in state_dict.items():
        # 跳过非权重参数
        if not isinstance(value, torch.Tensor):
            continue
        
        # 转换为numpy
        param_np = value.detach().cpu().numpy()
        
        # 转换参数名称
        ms_key = key
        
        # 处理特殊的层类型
        if 'patch_embed' in key:
            # patch embedding层
            if 'proj.weight' in key:
                # Conv2d权重: (C_out, C_in, H, W) -> (C_out, C_in, H, W)
                ms_key = key.replace('patch_embed.proj', 'patch_embed.conv')
            elif 'proj.bias' in key:
                ms_key = key.replace('patch_embed.proj', 'patch_embed.conv')
        
        elif 'blocks' in key:
            # Transformer块
            # layernorm参数
            if 'norm1.weight' in key:
                ms_key = key.replace('norm1.weight', 'layernorm1.gamma')
            elif 'norm1.bias' in key:
                ms_key = key.replace('norm1.bias', 'layernorm1.beta')
            elif 'norm2.weight' in key:
                ms_key = key.replace('norm2.weight', 'layernorm2.gamma')
            elif 'norm2.bias' in key:
                ms_key = key.replace('norm2.bias', 'layernorm2.beta')
            
            # attention参数
            elif 'attn.qkv.weight' in key:
                ms_key = key.replace('attn.qkv.weight', 'attention.wqkv.weight')
            elif 'attn.qkv.bias' in key:
                ms_key = key.replace('attn.qkv.bias', 'attention.wqkv.bias')
            elif 'attn.proj.weight' in key:
                ms_key = key.replace('attn.proj.weight', 'attention.wo.weight')
            elif 'attn.proj.bias' in key:
                ms_key = key.replace('attn.proj.bias', 'attention.wo.bias')
            
            # MLP参数
            elif 'mlp.fc1.weight' in key:
                ms_key = key.replace('mlp.fc1.weight', 'mlp.w1.weight')
            elif 'mlp.fc1.bias' in key:
                ms_key = key.replace('mlp.fc1.bias', 'mlp.w1.bias')
            elif 'mlp.fc2.weight' in key:
                ms_key = key.replace('mlp.fc2.weight', 'mlp.w2.weight')
            elif 'mlp.fc2.bias' in key:
                ms_key = key.replace('mlp.fc2.bias', 'mlp.w2.bias')
        
        elif 'head' in key:
            # 分类头（通常不需要）
            continue
        
        # 转置权重（如果需要）
        if 'weight' in key and param_np.ndim == 2:
            # Linear层权重: (out, in) -> (in, out) for MindSpore
            param_np = param_np.T
        
        ms_params[ms_key] = param_np
    
    print(f"    转换了 {len(ms_params)} 个参数")
    
    # 保存MindSpore格式
    ms_ckpt_path = os.path.join(output_dir, f"{model_name.lower().replace('-', '_')}.ckpt")
    
    print(f"    保存到: {ms_ckpt_path}")
    
    # 方法1：使用save_checkpoint
    try:
        # 创建参数列表
        param_list = []
        for name, param in ms_params.items():
            param_list.append({'name': name, 'data': ms.Tensor(param)})
        
        # 保存
        ms.save_checkpoint(param_list, ms_ckpt_path)
        print(f"\n✓ 转换完成！")
        print(f"✓ MindSpore模型已保存: {ms_ckpt_path}")
        
    except Exception as e:
        print(f"    保存失败: {e}")
        
        # 备选方法：保存为numpy
        print("\n    尝试备选方法...")
        npz_path = os.path.join(output_dir, f"{model_name.lower().replace('-', '_')}_weights.npz")
        np.savez(npz_path, **ms_params)
        print(f"    ✓ 权重已保存为numpy: {npz_path}")
        
        # 创建简单的checkpoint文件
        print("    创建checkpoint文件...")
        create_simple_checkpoint(ms_params, ms_ckpt_path)


def create_simple_checkpoint(params_dict, save_path):
    """创建简单的checkpoint文件"""
    import mindspore as ms
    
    # 创建参数列表
    param_list = []
    for name, param in params_dict.items():
        param_list.append({'name': name, 'data': ms.Tensor(param)})
    
    # 保存
    ms.save_checkpoint(param_list, save_path)
    print(f"✓ Checkpoint已保存: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  UNI模型下载与转换工具")
    print("=" * 50)
    print("\n功能：")
    print("  1. 从HuggingFace下载PyTorch模型")
    print("  2. 转换为MindSpore格式(.ckpt)")
    print()
    
    download_and_convert()
