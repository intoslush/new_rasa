import torch

class MLMMixin:
    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        device = input_ids.device

        # 概率矩阵与 input_ids 在同一设备
        if probability_matrix is None:
            prob = torch.full(input_ids.shape, self.mlm_probability, device=device, dtype=torch.float32)
        else:
            prob = probability_matrix.to(device=device, dtype=torch.float32)

        if masked_indices is None:
            masked_indices = torch.bernoulli(prob).to(dtype=torch.bool, device=device)

        # 不 mask PAD/CLS/SEP
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices[input_ids == self.tokenizer.sep_token_id] = False

        if targets is not None:
            targets = targets.to(device)
            targets[~masked_indices] = -100  # 只对被 mask 的位置算 loss

        # 80%：换成 [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8, device=device, dtype=torch.float32)
        ).to(torch.bool) & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10%：换成随机词
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device, dtype=torch.float32))
            .to(torch.bool) & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
        input_ids[indices_random] = random_words[indices_random]

        # 剩下 10%：保留原词
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids