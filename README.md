# ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding

This repository contains the official implementation of **ASAP**, proposed in our paper accepted to **CVPR 2025**:  
[![arXiv](https://img.shields.io/badge/arXiv-2412.12718-B31B1B.svg)](https://arxiv.org/abs/2412.12718)

---

## 📖 Paper Overview

**ASAP** is a novel framework for **Detecting and Grounding Multi-modal Media Manipulations (DGM4)**.  
It focuses on improving **cross-modal semantic alignment** between images and text to enhance manipulation detection performance.

### 🔍 Key Highlights

- **Large Model-Assisted Alignment (LMA):**  
  Leverages auxiliary textual information (e.g., captions, explanations) generated by large pre-trained models to enhance semantic consistency between modalities.

- **Manipulation-Guided Cross Attention (MGCA):**  
  Enables the model to focus more accurately on manipulated regions via task-specific cross-modal attention mechanisms.

- **State-of-the-Art Performance:**  
  Extensive experiments demonstrate that ASAP significantly outperforms previous methods on multi-modal manipulation detection and grounding tasks.

<p align="center">
  <img src="./examples/framework.png" alt="ASAP Framework Overview" width="700"/>
</p>

---

## 🚀 Code Release

We have released the **first version** of the code in this repository.  
For dataset preparation and environment setup, please refer to the detailed instructions in:  
👉 [MultiModal-DeepFake (by rshaojimmy)](https://github.com/rshaojimmy/MultiModal-DeepFake)

---

## 📂 Acknowledgement

We sincerely thank the authors of [MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake) for their excellent work.  
We heavily used the code from their repository in developing this project.

---

## 📄 Citation  

If you find our work helpful, please cite our CVPR 2025 paper:

```bibtex
@inproceedings{zhang2025asap,
  title     = {{ASAP:} Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding},
  author    = {Zhenxing Zhang and Yaxiong Wang and Lechao Cheng and Zhun Zhong and Dan Guo and Meng Wang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  pages     = {To appear},
  url       = {https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_ASAP_Advancing_Semantic_Alignment_Promotes_Multi-Modal_Manipulation_Detecting_and_Grounding_CVPR_2025_paper.pdf}
}
