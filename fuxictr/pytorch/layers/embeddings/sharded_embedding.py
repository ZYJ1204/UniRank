# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


class ShardedEmbedding(nn.Module):
    """
    分片 Embedding 层，用于 DDP 分布式训练。
    
    将 embedding 表按行分片到不同的 GPU 上，减少内存占用。
    每个 rank 只存储对应分片的 embedding 参数。
    
    使用方式：
        - 在 DDP 模式下，用 ShardedEmbedding 替代 nn.Embedding
        - 前向传播时会自动处理 gather 操作
    
    Args:
        num_embeddings: embedding 表的总大小
        embedding_dim: embedding 维度
        padding_idx: padding index (default: None)
        distributed: 是否启用分布式模式 (default: False)
        rank: 当前 rank (default: 0)
        world_size: world size (default: 1)
    """
    
    def __init__(self, 
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 distributed: bool = False,
                 rank: int = 0,
                 world_size: int = 1):
        super(ShardedEmbedding, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        
        if distributed and world_size > 1:
            # 分片模式：每个 rank 只存储一部分 embedding
            self.shard_size = (num_embeddings + world_size - 1) // world_size
            self.shard_start = rank * self.shard_size
            self.shard_end = min(self.shard_start + self.shard_size, num_embeddings)
            self.local_num_embeddings = self.shard_end - self.shard_start
            
            # 创建本地 embedding 表（只包含分片部分）
            self.embedding = nn.Embedding(
                self.local_num_embeddings,
                embedding_dim,
                padding_idx=None  # 不使用 padding_idx，手动处理
            )
            self._padding_idx = padding_idx
        else:
            # 非分布式模式：标准 embedding
            self.shard_size = num_embeddings
            self.shard_start = 0
            self.shard_end = num_embeddings
            self.local_num_embeddings = num_embeddings
            
            self.embedding = nn.Embedding(
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx
            )
            self._padding_idx = None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            input: 输入 indices，shape 为 [*, ]
            
        Returns:
            embedding 结果，shape 为 [*, embedding_dim]
        """
        if not self.distributed or self.world_size == 1:
            # 非分布式模式，直接返回
            return self.embedding(input)
        
        # 分布式模式：需要处理 gather
        input_shape = input.shape
        input_flat = input.reshape(-1)
        
        # 创建输出张量
        output_shape = list(input_shape) + [self.embedding_dim]
        output = torch.zeros(output_shape, dtype=torch.float32, device=input.device)
        output_flat = output.reshape(-1, self.embedding_dim)
        
        # 对每个 rank 的分片进行查询
        for shard_rank in range(self.world_size):
            shard_start = shard_rank * self.shard_size
            shard_end = min(shard_start + self.shard_size, self.num_embeddings)
            
            # 找到属于这个分片的 indices
            mask = (input_flat >= shard_start) & (input_flat < shard_end)
            
            if mask.any():
                local_indices = input_flat[mask] - shard_start
                
                if shard_rank == self.rank:
                    # 本地查询
                    embeddings = self.embedding(local_indices.long())
                else:
                    # 远程查询：通过 broadcast 获取
                    if self.rank == shard_rank:
                        embeddings = self.embedding(local_indices.long())
                    else:
                        embeddings = torch.zeros(
                            (local_indices.shape[0], self.embedding_dim),
                            dtype=torch.float32,
                            device=input.device
                        )
                    
                    # Broadcast 从 shard_rank 到所有 ranks
                    dist.broadcast(embeddings, src=shard_rank)
                
                # 填充到输出中
                output_flat[mask] = embeddings
        
        # 处理 padding_idx
        if self._padding_idx is not None:
            padding_mask = (input_flat == self._padding_idx)
            if padding_mask.any():
                output_flat[padding_mask] = 0
        
        return output.reshape(output_shape)


class ShardedEmbeddingByFeature(nn.Module):
    """
    按特征维度分片的 Embedding（替代方案）。
    
    如果 embedding 维度较大，可以按特征维度分片。
    例如：embedding_dim=100，world_size=4，每个 rank 负责 25 维。
    
    Args:
        num_embeddings: embedding 表大小
        embedding_dim: embedding 维度
        padding_idx: padding index
        distributed: 是否启用分布式
        rank: 当前 rank
        world_size: world size
    """
    
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 distributed: bool = False,
                 rank: int = 0,
                 world_size: int = 1):
        super(ShardedEmbeddingByFeature, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        
        if distributed and world_size > 1:
            # 按维度分片
            self.feature_shard_size = (embedding_dim + world_size - 1) // world_size
            self.feature_shard_start = rank * self.feature_shard_size
            self.feature_shard_end = min(
                self.feature_shard_start + self.feature_shard_size,
                embedding_dim
            )
            self.local_embedding_dim = self.feature_shard_end - self.feature_shard_start
            
            # 创建本地 embedding（全部行，部分列）
            self.embedding = nn.Embedding(
                num_embeddings,
                self.local_embedding_dim,
                padding_idx=padding_idx
            )
        else:
            # 非分布式模式
            self.feature_shard_size = embedding_dim
            self.feature_shard_start = 0
            self.feature_shard_end = embedding_dim
            self.local_embedding_dim = embedding_dim
            
            self.embedding = nn.Embedding(
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            input: 输入 indices
            
        Returns:
            embedding 结果
        """
        if not self.distributed or self.world_size == 1:
            return self.embedding(input)
        
        # 本地 embedding 结果
        local_output = self.embedding(input)
        
        # 通过 all-gather 合并所有 ranks 的结果
        output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output)
        
        # 在特征维度上拼接
        output = torch.cat(output_list, dim=-1)
        
        # 截断到正确的维度（处理不整除的情况）
        output = output[..., :self.embedding_dim]
        
        return output
