import json
import os
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from collections import defaultdict
from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, weak_pos_pair_probability=0.1):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.weak_pos_pair_probability = weak_pos_pair_probability

        self.pairs = []
        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)
        person_id2idx = {}
        person_idx = 0

        for ann in anns:
            pid = ann['id']
            if pid not in person_id2idx:
                person_id2idx[pid] = person_idx
                person_idx += 1
            idx = person_id2idx[pid]
            self.person2image[idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, idx))
                self.person2text[idx].append(cap)

        self.pseudo_labels = [-1] * len(self.pairs)
        self.valid_indices = list(range(len(self.pairs)))
        self.mode = "train"

    def set_pseudo_labels(self, labels):
        assert len(labels) == len(self.pairs), "标签数量与样本数量不一致"
        print("成功将伪标签写入数据集中")
        self.pseudo_labels = labels
        self.valid_indices = [i for i, label in enumerate(labels) if label != -1]

    def __len__(self):
        return len(self.valid_indices) if self.mode == 'train' and self.pseudo_labels else len(self.pairs)

    def augment(self, caption, pid):
        if np.random.rand() < self.weak_pos_pair_probability:
            aug_caption = np.random.choice(self.person2text[pid], 1).item()
            replaced = 1
        else:
            aug_caption = caption
            replaced = 0
        return aug_caption, replaced

    def __getitem__(self, idx):
        """
        person	数据集中原始ID编号,如：第一个人是 0，第二个是 1
        pseudo_labels[real_idx]	动态生成的 int 或 -1,如：DBSCAN 聚类后为：13、17、22
        real_idx	Dataset 的真实下标,如：第 127 个样本，real_idx = 127
        """ 
        if self.mode == 'train' and self.pseudo_labels:
            real_idx = self.valid_indices[idx]
        else:
            real_idx = idx

        image_path, caption, pid = self.pairs[real_idx]
        aug_caption, replaced = self.augment(caption, pid)

        full_image_path = os.path.join(self.image_root, image_path)
        img = Image.open(full_image_path).convert('RGB')
        img = self.transform(img)

        caption_tokens = pre_caption(caption, self.max_words)
        mlm_tokens = pre_caption(aug_caption, self.max_words)
        mlm_labels = replaced

        ret = {
            'pids': pid,
            'image_ids': real_idx,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }
        return ret


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

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['file_path'])
            pid = ann['id']
            self.img2person.append(pid)
            for cap in ann['captions']:
                self.text.append(pre_caption(cap, self.max_words))
                self.txt2person.append(pid)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        ann = self.ann[idx]
        image_path = os.path.join(self.image_root, ann['file_path'])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        ret = {
            'pids': ann['id'],
            'image_ids': idx,
            'images': img,
            'caption_ids': None,    # 无 caption（图像评估用）
            'mlm_ids': None,
            'mlm_labels': None
        }
        return ret
