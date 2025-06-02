from distutils.command.config import config
import json
import os
import random

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand

class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True): 
        
        self.root_dir = '../data/rshaojimmy'       
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train

        if is_train:
            self.img_cap =json.load(open('caption_train.json','r'))
            self.text_prompt = json.load(open('dataset/prompt_engine.json'))
            self.text_prompt_mistral = json.load(open('Prompt_Generate/prompt_train.json'))

        elif ann_file[0].endswith('test.json'):
            self.img_cap = {}
            self.text_prompt = {}
            self.text_prompt_mistral = {}
        else:
            self.img_cap = {}
            self.text_prompt = {}
            self.text_prompt_mistral = {}
        
    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    
    def get_res_pos(self,x, y, w, h):
        patch = np.full((256,), -1,dtype=float)

        block_size = 16
        num_blocks = 256 // block_size
        x_coords, y_coords = np.meshgrid(np.arange(0, 256, block_size), np.arange(0, 256, block_size))
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        x_coords_end = x_coords + block_size
        y_coords_end = y_coords + block_size

        overlap = (
            (x_coords < x + w) & (x_coords_end > x) & 
            (y_coords < y + h) & (y_coords_end > y)
        )
        patch[overlap] = 1
        patch_reshaped = patch.reshape((num_blocks, num_blocks))
        adjacency = np.zeros_like(patch_reshaped, dtype=bool)
        
        adjacency[:-1, :] |= patch_reshaped[1:, :] == 1
        adjacency[1:, :] |= patch_reshaped[:-1, :] == 1
        adjacency[:, :-1] |= patch_reshaped[:, 1:] == 1
        adjacency[:, 1:] |= patch_reshaped[:, :-1] == 1

        patch_reshaped[(patch_reshaped == -1) & adjacency] = 0
        patch = patch_reshaped.flatten()
        patch_tensor = torch.tensor(patch, dtype=torch.float)
        # patch_matrix = patch.reshape((16, 16))

        # print(patch_matrix)
        return patch_tensor
    def get_res_pos_Attention(self,x, y, w, h):
        patch = np.full((256,), 0,dtype=float)

        block_size = 16
        num_blocks = 256 // block_size
        x_coords, y_coords = np.meshgrid(np.arange(0, 256, block_size), np.arange(0, 256, block_size))
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        x_coords_end = x_coords + block_size
        y_coords_end = y_coords + block_size

        overlap = (
            (x_coords < x + w) & (x_coords_end > x) & 
            (y_coords < y + h) & (y_coords_end > y)
        )
        patch[overlap] = 1
        # patch_reshaped = patch.reshape((num_blocks, num_blocks))
        # adjacency = np.zeros_like(patch_reshaped, dtype=bool)
        
        # adjacency[:-1, :] |= patch_reshaped[1:, :] == 1
        # adjacency[1:, :] |= patch_reshaped[:-1, :] == 1
        # adjacency[:, :-1] |= patch_reshaped[:, 1:] == 1
        # adjacency[:, 1:] |= patch_reshaped[:, :-1] == 1

        # patch_reshaped[(patch_reshaped == -1) & adjacency] = 0
        # patch = patch_reshaped.flatten()
        patch_tensor = torch.tensor(patch, dtype=torch.float)
        # patch_matrix = patch.reshape((16, 16))
        # print(patch_matrix)
        return patch_tensor
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                         
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
            res_fake_pos = torch.full((256,), 0,dtype=torch.float)
            res_fake_pos_patch = torch.full((256,), -1,dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h
            res_fake_pos = self.get_res_pos_Attention(x,y,w,h)
            res_fake_pos_patch = self.get_res_pos(x,y,w,h)
            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words)
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1

        #----------------------------------------------------------

        real_cap = self.img_cap.get(img_dir, caption)
        real_cap = pre_caption(real_cap, self.max_words)

        real_prom_mistral = self.text_prompt_mistral.get(img_dir, caption)
        real_prom_mistral = pre_caption(real_prom_mistral, self.max_words)

        real_prom = self.text_prompt.get(img_dir, real_prom_mistral)
        real_prom = pre_caption(real_prom, self.max_words)
                
        return image, label, caption, fake_image_box, fake_text_pos_list, W, H, real_cap, real_prom, res_fake_pos, res_fake_pos_patch
