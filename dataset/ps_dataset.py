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
    def __init__(self, ann_file, transform, image_root, max_words=30, weak_pos_pair_probability=0.1):
        anns = []
        for f in ann_file:
            anns += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.weak_pos_pair_probability = weak_pos_pair_probability  # 待修改
        self.person2image = defaultdict(list)
        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []
        for ann in anns:
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            self.person2image[person_idx].append(ann['file_path'])
            for cap in ann['captions']:
                self.pairs.append((ann['file_path'], cap, person_idx))
                self.person2text[person_idx].append(cap)
        
        self.pseudo_labels = [-1] * len(self.pairs)
        self.valid_indices = list(range(len(self.pairs)))
        self.mode = "train"
        
    def set_pseudo_labels(self, labels):
        assert len(labels) == len(self.pairs), "标签数量和样本数量不一致"
        print("成功将伪标签写入数据集中")
        self.pseudo_labels = labels
        self.valid_indices = [i for i, label in enumerate(labels) if label != -1]

    def __len__(self):
        if self.mode == 'train' and self.pseudo_labels is not None:
            return len(self.valid_indices)
        else:
            return len(self.pairs)

    def augment(self, caption, person):
        caption_aug = caption
        if np.random.random() < self.weak_pos_pair_probability:
            caption_aug = np.random.choice(self.person2text[person], 1).item()
        if caption_aug == caption:
            replace = 0
        else:
            replace = 1
        return caption_aug, replace

    def __getitem__(self, index):
        """
        person	数据集中原始ID编号,如：第一个人是 0，第二个是 1
        pseudo_labels[real_idx]	动态生成的 int 或 -1,如：DBSCAN 聚类后为：13、17、22
        real_idx	Dataset 的真实下标,如：第 127 个样本，real_idx = 127
        """ 
        
        if self.mode == 'train' and self.pseudo_labels is not None:
            real_idx = self.valid_indices[index]
            if index >= len(self.valid_indices):
                raise IndexError(f"Index {index} out of range: valid_indices has length {len(self.valid_indices)}")
        else:
            real_idx = index
        image_path, caption, person = self.pairs[real_idx]
        caption_aug, replace = self.augment(caption, person)
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
