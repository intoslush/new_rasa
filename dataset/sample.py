from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch
import math

class ValidIndexDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.epoch = 0
        self.valid_indices = self._sync_valid_indices()

    def _sync_valid_indices(self):
        """
        获取并同步 valid_indices，避免多进程不一致。
        """
        # 主进程获取
        if dist.get_rank() == 0:
            if hasattr(self.dataset, 'valid_indices'):
                valid_indices = self.dataset.valid_indices
            else:
                valid_indices = list(range(len(self.dataset)))
        else:
            valid_indices = None

        # 广播同步
        object_list = [valid_indices]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]
    def set_valid_indices(self, valid_indices):
        self.valid_indices = valid_indices
        
        
    def __iter__(self):

        indices = self.valid_indices

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # 统一长度逻辑
        if self.drop_last:
            num_samples_per_replica = len(indices) // self.num_replicas
            total_size = num_samples_per_replica * self.num_replicas
            indices = indices[:total_size]
        else:
            num_samples_per_replica = int(math.ceil(len(indices) / self.num_replicas))
            total_size = num_samples_per_replica * self.num_replicas
            padding_size = total_size - len(indices)
            if padding_size > 0:
                # 重复填充（防止部分 rank 拿不到数据）
                indices += indices[:padding_size]

        assert len(indices) == total_size

        # 分配给当前 rank
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == num_samples_per_replica

        return iter(indices)

    def __len__(self):
        if self.drop_last:
            return len(self.valid_indices) // self.num_replicas
        else:
            return int(math.ceil(len(self.valid_indices) / self.num_replicas))

    def set_epoch(self, epoch):
        self.epoch = epoch