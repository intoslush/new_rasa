import json
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from collections import defaultdict
from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ps_train_dataset(Dataset):
    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        max_words=30,
        weak_pos_pair_probability=0.1,
        # 新增：伪标签增广的触发概率（默认与 weak_pos_pair_probability 一致更好理解）
        pseudo_pos_pair_probability=None,
        # 新增：增广策略 original | pseudo | none
        augment_policy: str = "none",
    ):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.weak_pos_pair_probability = float(weak_pos_pair_probability)
        self.pseudo_pos_pair_probability = (
            float(pseudo_pos_pair_probability)
            if pseudo_pos_pair_probability is not None
            else float(weak_pos_pair_probability)
        )
        assert augment_policy in ("original", "pseudo", "none"), "augment_policy must be one of ['original','pseudo','none']"
        self.augment_policy = augment_policy

        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)

        person_id2idx = {}
        n = 0
        self.pairs = []  # list of (file_path, caption, person_idx)

        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx:
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person2image[person_idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, person_idx))
                self.person2text[person_idx].append(cap)

        # 伪标签相关
        self.pseudo_labels = [-1] * len(self.pairs)
        self.valid_indices = list(range(len(self.pairs)))
        self.cluster2indices = defaultdict(list)  # 伪标签簇 -> [real_idx,...]

        self.mode = "train"

    # ========== 新增/增强：API ==========
    def set_augment_policy(self, policy: str):
        """切换增广策略：'original' | 'pseudo' | 'none'"""
        assert policy in ("original", "pseudo", "none")
        self.augment_policy = policy

    def set_pseudo_labels(self, labels):
        """
        每个 epoch 后调用。会更新:
        - self.pseudo_labels
        - self.valid_indices
        - self.cluster2indices 供 pseudo augment 使用
        """
        assert len(labels) == len(self.pairs), f"标签数量{len(labels)}和样本数量不一致{len(self.pairs)}"
        print("成功将伪标签写入数据集中")
        self.pseudo_labels = labels
        self.valid_indices = [i for i, label in enumerate(labels) if label != -1]

        # 重建伪标签簇索引（排除 -1）
        self.cluster2indices.clear()
        for idx, c in enumerate(labels):
            if c != -1:
                self.cluster2indices[c.item()].append(idx)




    def set_probs(self, weak_pos_pair_probability=None, pseudo_pos_pair_probability=None):
        """可选：动态调整两种增广触发概率"""
        if weak_pos_pair_probability is not None:
            self.weak_pos_pair_probability = float(weak_pos_pair_probability)
        if pseudo_pos_pair_probability is not None:
            self.pseudo_pos_pair_probability = float(pseudo_pos_pair_probability)

    # ========== 原始长度逻辑保持 ==========
    def __len__(self):
        if self.mode == 'train' and self.pseudo_labels is not None:
            return len(self.valid_indices)
        else:
            return len(self.pairs)

    # ========== 增广实现 ==========
    def _augment_person(self, caption, person):
        """原始增广逻辑：在同 person 的 caption 中随机替换"""
        caption_aug = caption
        if self.weak_pos_pair_probability > 0 and np.random.random() < self.weak_pos_pair_probability:
            # 注意：如果 person2text[person] 只有一个元素，np.random.choice 仍会返回它本身
            caption_aug = np.random.choice(self.person2text[person], 1).item()
        replace = 1 if caption_aug != caption else 0
        return caption_aug, replace

    def _augment_pseudo(self, caption, real_idx):
        """
        新增：基于伪标签簇的增广。
        从与 real_idx 同簇（相同 pseudo_label）的其它样本里采样 caption。
        """
        caption_aug = caption
        replace = 0

        if self.pseudo_pos_pair_probability <= 0:
            return caption_aug, replace
        if np.random.random() >= self.pseudo_pos_pair_probability:
            return caption_aug, replace

        # 取当前样本的伪标签
        c = self.pseudo_labels[real_idx].item() if self.pseudo_labels is not None else -1
        if c == -1:
            return caption_aug, replace  # 无簇/噪声簇，跳过

        candidates = self.cluster2indices.get(c, [])
        if not candidates or (len(candidates) == 1 and candidates[0] == real_idx):
            return caption_aug, replace  # 同簇没有其他样本

        # 从同簇里选一个 != real_idx 的样本，拿它的 caption
        if len(candidates) > 1:
            # 过滤自身
            pool = [j for j in candidates if j != real_idx]
        else:
            pool = candidates

        if not pool:
            return caption_aug, replace

        j = np.random.choice(pool, 1).item()
        caption_aug = self.pairs[j][1]  # 使用同簇样本的 caption
        replace = 1 if caption_aug != caption else 0
        return caption_aug, replace

    def augment(self, caption, person, real_idx=None):
        """
        统一入口：根据 augment_policy 选择增广来源。
        - original: 同 person 替换
        - pseudo:   同伪标签簇替换
        - none:     不做替换
        """
        if self.augment_policy == "none":
            return caption, 0

        if self.augment_policy == "pseudo":
            # 基于伪标签的增广
            return self._augment_pseudo(caption, real_idx)

        # 否则走原始逻辑
        return self._augment_person(caption, person)

    # ========== 取样保持你的输出结构 ==========
    def __getitem__(self, index):
        """
        person  数据集中原始ID编号,如：第一个人是 0，第二个是 1
        pseudo_labels[real_idx] 动态生成的 int 或 -1,如：DBSCAN 聚类后为：13、17、22
        real_idx    Dataset 的真实下标,如：第 127 个样本，real_idx = 127
        """ 
        if self.mode == 'train' and self.pseudo_labels is not None:
            real_idx = self.valid_indices[index]
            if index >= len(self.valid_indices):
                raise IndexError(f"Index {index} out of range: valid_indices has length {len(self.valid_indices)}")
        else:
            real_idx = index

        image_path, caption, person = self.pairs[real_idx]

        # 关键：把 real_idx 传入增广，支持 pseudo augment
        caption_aug, replace = self.augment(caption, person, real_idx=real_idx)

        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)
        image2 = self.transform(image)

        caption1 = pre_caption(caption, self.max_words)
        caption2 = pre_caption(caption_aug, self.max_words)

        return {
            'image1': image1,
            'image2': image2,
            'caption1': caption1,
            'caption2': caption2,
            'person_id': person,
            'replace_flag': replace,
            'real_index': real_idx,
            'pseudo_label': self.pseudo_labels[real_idx]
        }

class ps_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []
        person2img = defaultdict(list)
        person2txt = defaultdict(list)
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['file_path'])
            person_id = ann['id']
            person2img[person_id].append(img_id)
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, self.max_words))
                person2txt[person_id].append(txt_id)
                self.txt2person.append(person_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['file_path'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        # return image, index
        return {
                'image': image_tensor,
                'index': index,
            }
