from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_
class Multilmodelattention(nn.Module):
    def __init__(self,input_dim,head,*args,**kwargs):
        super().__init__()
        self.selfatten = nn.MultiheadAttention(input_dim,head,dropout=0.0, batch_first=True)
        # self.crossatten = nn.MultiheadAttention(input_dim,head,dropout=0.0, batch_first=True)
        
    def forward(self, query,key,value):
        x, _ = self.selfatten(query, query, query)

        # x, _ = self.crossatten(x, key, value, key_padding_mask=key_padding_mask)
        x = x + query
        return x
class HAMMER(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True
                 ):
        super().__init__()
        
        # self.args = args
        # self.tokenizer = tokenizer 
        # embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        text_width = 768
        # self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)         

        # self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        # self.queue_size = config['queue_size']
        # self.momentum = config['momentum']  

        # creat itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)
        # self.itm_head_m = self.build_mlp(input_dim=text_width, output_dim=2)
        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)
        self.patch_head = nn.Linear(text_width,1)
        # creat multi-cls head
        # self.cls_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # create momentum models
        # self.visual_encoder_m = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertForTokenClassification.from_pretrained(text_encoder, 
        #                                                             config=bert_config,
        #                                                             label_smoothing=config['label_smoothing'])       
        # self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        # self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                     [self.vision_proj,self.vision_proj_m],
        #                     [self.text_encoder,self.text_encoder_m],
        #                     [self.text_proj,self.text_proj_m],
        #                    ]
        
        # self.copy_params()

        # create the queue
        # self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("cap_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("prom_queue", torch.randn(embed_dim, self.queue_size))  
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        # self.cap_queue = nn.functional.normalize(self.text_queue, dim=0)
        # self.prom_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr =nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        # self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.it_cross_attn = Multilmodelattention(text_width, 12)
        #--------------VIKI---------------
        # self.UtilsC = nn.Parameter(torch.tensor(0.5))
        self.UtilsB = nn.Parameter(torch.tensor(0.5))
        #---------------------------------
        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes
    def get_pre_patch_loss(self,pre_patch,gt_patch):
        criterion = nn.BCEWithLogitsLoss()
        mask = (gt_patch != -1).float()

        masked_pre_patch = pre_patch * mask
        masked_gt_patch = gt_patch * mask

        loss = criterion(masked_pre_patch, masked_gt_patch)
        return loss
    def get_pre_Att_loss(self,pre_patch,gt_patch):
        criterion = nn.BCEWithLogitsLoss()

        loss = criterion(pre_patch, gt_patch)
        return loss
    def forward(self, image, label, text, fake_image_box, fake_text_pos, cap_input,prom_input,res_fake_pos,res_fake_pos_patch, alpha=0, is_train=True):
        if is_train:
            # with torch.no_grad():
            #     self.temp.clamp_(0.001,0.5)
            ##================= multi-label convert ========================## 
            multicls_label, real_label_pos = get_multi_label(label, image)
            
            ##================= MAC ========================## 
            image_embeds = self.visual_encoder(image) 

            with torch.no_grad():
                bs = image.size(0)     
            # local features of visual part
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn = self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(image_embeds), 
                                              value=self.norm_layer_it_cross_atten(image_embeds))
            
            patch_pre = self.patch_head(self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))
            loss_Patch = self.get_pre_patch_loss(patch_pre.squeeze(2),res_fake_pos_patch)

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            ##================= BIC ========================## 
            # forward the positve image-text pair
            # output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
            #                                 attention_mask = text.attention_mask,
            #                                 encoder_hidden_states = image_embeds,
            #                                 encoder_attention_mask = image_atts, 
            #                                 output_attentions = True,     
            #                                 return_dict = True,
            #                                 mode = 'fusion',
            #                             )       
            # text_words = text_embeds.size(1)
            # attentionMapGT = res_fake_pos.unsqueeze(-1).repeat(1,1,text_words)  
            # for pos in fake_text_pos:
            #     attentionMapGT[:,:,pos] = 1
            # # print(output_pos.cross_attentions[0,1,:,:])
            # pre_attentions_map = output_pos.cross_attentions.mean(dim=1).permute(0,2,1)

            # loss = self.get_pre_Att_loss(pre_attentions_map[:,1:,:],attentionMapGT)
            real_image_label_pos = []
            orig = np.where(np.array(label)=='orig')[0].tolist()
            text_swrap_only = np.where(np.array(label) == 'text_swap')[0].tolist()
            text_attr_only = np.where(np.array(label) == 'text_attribute')[0].tolist()
            real_image_label_pos.extend(orig)
            real_image_label_pos.extend(text_swrap_only)
            real_image_label_pos.extend(text_attr_only)
            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)

            itm_labels[real_image_label_pos] = 0 # fine-grained matching: only orig should be matched, 0 here means img-text matching
            vl_output = self.itm_head(self.UtilsB * local_feat_aggr.squeeze(1))   
            loss_BIC = F.cross_entropy(vl_output, itm_labels) 
            
            LossAnother =  0.1*loss_Patch
            return loss_BIC, loss_bbox, loss_giou, LossAnother

        else:
            image_embeds = self.visual_encoder(image) 
            # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            # text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
            #                                 return_dict = True, mode = 'text')            
            # text_embeds = text_output.last_hidden_state

            # # forward the positve image-text pair
            # output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
            #                                 attention_mask = text.attention_mask,
            #                                 encoder_hidden_states = image_embeds,
            #                                 encoder_attention_mask = image_atts, 
            #                                 output_attentions = True,     
            #                                 return_dict = True,
            #                                 mode = 'fusion',
            #                             )               
            ##================= IMG ========================## 
            bs = image.size(0)
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            # text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            # local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn =  self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(image_embeds), 
                                              value=self.norm_layer_it_cross_atten(image_embeds))

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            ##================= BIC ========================## 
            logits_real_fake = self.itm_head(self.UtilsB * local_feat_aggr.squeeze(1))
            ##================= MLC ========================## 
            # logits_multicls = self.cls_head(output_pos.last_hidden_state[:,0,:] + self.UtilsC * local_feat_aggr.squeeze(1))
            ##================= TMG ========================##   
            # input_ids = text.input_ids.clone()
            # logits_tok = self.text_encoder(input_ids, 
            #                             attention_mask = text.attention_mask,
            #                             encoder_hidden_states = image_embeds,
            #                             encoder_attention_mask = image_atts,      
            #                             return_dict = True,
            #                             return_logits = True,   
            #                             )     
            return logits_real_fake, output_coord   


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, cap_feat,prom_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        cap_feats = concat_all_gather(cap_feat)
        prom_feats = concat_all_gather(prom_feat)
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.cap_queue[:, ptr:ptr + batch_size] = cap_feats.T
        self.prom_queue[:, ptr:ptr + batch_size] = prom_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
