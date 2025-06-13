import os
import sys
sys.path.append('../VisCPM/')

import pdb
#from torch.utils.data import Dataset
import json
#from torch.utils.data import DataLoader
from tqdm import tqdm

from VisCPM import VisCPMChat
from PIL import Image
import torch

model_path = '../VisCPM/pytorch_model.bin'

'''
class DGM4_Dataset(Dataset):
    def __init__(self, ann_file='../DGM4/metadata/train.json'): 
        
        self.root_dir = '../'       
        self.ann = []
        self.ann = json.load(open(ann_file,'r'))
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                                
        return {'image':image, 'path': image_dir_all}

'''
'''dataset = DGM4_Dataset(ann_file='../DGM4/metadata/train.json')

loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )  
'''
split = 'test'
ann_file=f'../DGM4/metadata/{split}.json'
ann_list = json.load(open(ann_file,'r'))

#device = torch.device('cuda:1')
viscpm_chat = VisCPMChat(model_path, image_safety_checker=False)
root_dir = '../' 

results = {}
print(len(ann_list))
for i, data in tqdm(enumerate(ann_list)):
    #image, path = data['image'], data['path']

    img_dir = data['image']
        
    image_dir_all = f'{root_dir}/{img_dir}'
    image = Image.open(image_dir_all).convert('RGB')       
    
    prompt = 'Give the caption of this picture'
    answer, context, vis_hidden = viscpm_chat.chat(image, prompt)
    
    results[img_dir] = answer

json.dump(results, open(f'caption_{split}.json', 'w'))
