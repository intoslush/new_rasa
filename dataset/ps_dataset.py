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

def _to_int(x):
    # 兼容 torch scalar / numpy scalar / python int
    return int(x.item()) if hasattr(x, "item") else int(x)


class ps_train_dataset(Dataset):
    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        max_words=30,
        weak_pos_pair_probability=0.1,
        pseudo_pos_pair_probability=None,
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
        assert augment_policy in ("original", "pseudo", "none"), \
            "augment_policy must be one of ['original','pseudo','none']"
        self.augment_policy = augment_policy

        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)

        person_id2idx = {}
        n = 0

        # 新增：为每个唯一图像(file_path)分配一个 image_id
        self.image_path2id = {}
        self.id2image_path = []  # 可选：需要反查时用

        # 修改：pairs 存 (file_path, caption, person_idx, image_id)
        self.pairs = []

        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx:
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]

            img_rel_path = ann['file_path']
            self.person2image[person_idx].append(img_rel_path)

            # 分配 image_id（同一张图的多个 caption 共享同一个 image_id）
            if img_rel_path not in self.image_path2id:
                self.image_path2id[img_rel_path] = len(self.id2image_path)
                self.id2image_path.append(img_rel_path)
            image_id = self.image_path2id[img_rel_path]

            for cap in ann['captions']:
                self.pairs.append((img_rel_path, cap, person_idx, image_id))
                self.person2text[person_idx].append(cap)

        # 伪标签相关
        self.pseudo_labels = [-1] * len(self.pairs)
        self.valid_indices = list(range(len(self.pairs)))
        self.cluster2indices = defaultdict(list)

        self.mode = "train"

    def set_augment_policy(self, policy: str):
        assert policy in ("original", "pseudo", "none")
        self.augment_policy = policy

    def set_pseudo_labels(self, labels):
        assert len(labels) == len(self.pairs), \
            f"标签数量{len(labels)}和样本数量不一致{len(self.pairs)}"
        print("成功将伪标签写入数据集中")

        # 建议：统一转成 python int，避免后面 .item() / 类型不一致
        self.pseudo_labels = [_to_int(x) for x in labels]

        self.valid_indices = [i for i, label in enumerate(self.pseudo_labels) if label != -1]

        self.cluster2indices.clear()
        for idx, c in enumerate(self.pseudo_labels):
            if c != -1:
                self.cluster2indices[c].append(idx)

    def set_probs(self, weak_pos_pair_probability=None, pseudo_pos_pair_probability=None):
        if weak_pos_pair_probability is not None:
            self.weak_pos_pair_probability = float(weak_pos_pair_probability)
        if pseudo_pos_pair_probability is not None:
            self.pseudo_pos_pair_probability = float(pseudo_pos_pair_probability)

    def __len__(self):
        if self.mode == 'train' and self.pseudo_labels is not None:
            return len(self.valid_indices)
        else:
            return len(self.pairs)

    def _augment_person(self, caption, person):
        caption_aug = caption
        if self.weak_pos_pair_probability > 0 and np.random.random() < self.weak_pos_pair_probability:
            caption_aug = np.random.choice(self.person2text[person], 1).item()
        replace = 1 if caption_aug != caption else 0
        return caption_aug, replace

    def _augment_pseudo(self, caption, real_idx):
        caption_aug = caption
        replace = 0

        if self.pseudo_pos_pair_probability <= 0:
            return caption_aug, replace
        if np.random.random() >= self.pseudo_pos_pair_probability:
            return caption_aug, replace

        c = self.pseudo_labels[real_idx] if self.pseudo_labels is not None else -1
        if c == -1:
            return caption_aug, replace

        candidates = self.cluster2indices.get(c, [])
        if not candidates or (len(candidates) == 1 and candidates[0] == real_idx):
            return caption_aug, replace

        pool = [j for j in candidates if j != real_idx]
        if not pool:
            return caption_aug, replace

        j = np.random.choice(pool, 1).item()
        caption_aug = self.pairs[j][1]  # caption 仍然在 index=1
        replace = 1 if caption_aug != caption else 0
        return caption_aug, replace

    def augment(self, caption, person, real_idx=None):
        if self.augment_policy == "none":
            return caption, 0
        if self.augment_policy == "pseudo":
            return self._augment_pseudo(caption, real_idx)
        return self._augment_person(caption, person)

    def __getitem__(self, index):
        if self.mode == 'train' and self.pseudo_labels is not None:
            real_idx = self.valid_indices[index]
            if index >= len(self.valid_indices):
                raise IndexError(
                    f"Index {index} out of range: valid_indices has length {len(self.valid_indices)}"
                )
        else:
            real_idx = index

        # 修改：多取一个 image_id
        image_path, caption, person, image_id = self.pairs[real_idx]

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
            'image_id': image_id,              # 新增：同一张图的多条描述共享同一个 id
            'replace_flag': replace,
            'real_index': real_idx,
            'pseudo_label': self.pseudo_labels[real_idx]  # 现在是 python int / -1
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
