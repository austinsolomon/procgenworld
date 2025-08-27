---
license: mit
language:
- en
pipeline_tag: image-to-video
library_name: diffusers
base_model:
- Skywork/SkyReels-V2-I2V-1.3B-540P
---
# Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model
<div style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://github.com/SkyworkAI/Matrix-Game">
    <img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://arxiv.org/pdf/2508.13009">
    <img src="https://img.shields.io/badge/arXiv-Report-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="report">
  </a>
    <a href="https://matrix-game-v2.github.io/">
    <img src="https://img.shields.io/badge/Project%20Page-grey?style=flat&logo=huggingface&color=FFA500" alt="Project Page">
  </a>

  
</div>

## 📝 Overview
**Matrix-Game-2.0（1.8B）** is an interactive world model generates long videos on-the-fly via few-step auto-regressive diffusion 

## ✨ Key Features

- 🚀 **Feature 1**: **Real-Time Distillation**  Efficient ​​few-step diffusion​​ for streaming video synthesis at ​​25 FPS​​, producing ​​minute-level, high-fidelity videos​​ across complex environments with ultra-fast speed.
- 🖱️ **Feature 2**: **Precise Action Injection** A ​​mouse/keyboard-to-frame​​ module that embeds user inputs as direct interactions, enabling frame-level control and dynamic response in generated videos. 
- 🎬 **Feature 3**: **Massive Interactive Data Pipeline** A scalable production system for ​​Unreal Engine & GTA5​​ that generates ​​~1200 hours​​ of high-quality interactive video data, covering diverse scenes with frame-level realism.

## 🔥 Latest Updates

* [2025-08] 🎉 Initial release of Matrix-Game-2.0 Model

##  Model Overview
**Matrix-Game-2.0（1.8B）** is derived from the Wan. By removing the text branch and adding action modules, the model predicts next frames only from visual contents and corresponding actions.

![Model Overview](./architecture.png)

## 📈 Performance Comparison
### GameWorld Score Benchmark Comparison

| Model     | Image Quality ↑ | Aesthetic Quality ↑ | Temporal Cons. ↑ | Motion Smooth. ↑ | Keyboard Acc. ↑ | Mouse Acc. ↑ | Object Cons. | Scenario Cons.|
|-----------|------------------|-------------|-------------------|-------------------|------------------|---------------|-------------|-------------|
| Oasis     | 0.27             | 0.27        | 0.82              | **0.99**          | 0.73             | 0.56          | 0.18        |  **0.84** | 
| **Ours**  | **0.61**         | **0.50**    | **0.94**          | 0.98          | **0.91**         | **0.95**      | **0.64**    |  0.80    |

**Metric Descriptions**:

- **Image Quality** / **Aesthetic**: Visual fidelity and perceptual appeal of generated frames  
- **Temporal Consistency** / **Motion Smoothness**: Temporal coherence and smoothness between frames  
- **Keyboard Accuracy** / **Mouse Accuracy**: Accuracy in following user control signals  
- **Object Consistency**: Geometric stability and consistency of objects over time
- **Scenario Consistency**: Scenario consistency over time

  Please check our [GameWorld](https://github.com/SkyworkAI/Matrix-Game/tree/main/GameWorldScore) benchmark for detailed implementation.


## 🚀 Quick Start

```
# clone the repository:
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game/Matrix-Game-2

# install apex and FlashAttention
# Our project also depends on [FlashAttention](https://github.com/Dao-AILab/flash-attention)
# install dependencies:
pip install -r requirements.txt
python setup.py develop

# inference
python inference.py \
    --config_path configs/inference_yaml/{your-config}.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --img_path {path-to-the-input-image} \
    --output_folder outputs \
    --num_output_frames 150 \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder}
# inference streaming
python inference_streaming.py \
    --config_path configs/inference_yaml/{your-config}.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --output_folder outputs \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder}
```

## ⭐ Acknowledgements

We would like to express our gratitude to:

- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) for their strong base model
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [MineRL](https://github.com/minerllabs/minerl) for their excellent gym framework
- [Video-Pre-Training](https://github.com/openai/Video-Pre-Training) for their accurate Inverse Dynamics Model
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module 

We are grateful to the broader research community for their open exploration and contributions to the field of interactive world generation.

## 📎 Citation
If you find this project useful, please cite our paper:
```bibtex
  @article{he2025matrix,
    title={Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model},
    author={He, Xianglong and Peng, Chunli and Liu, Zexiang and Wang, Boyang and Zhang, Yifan and Cui, Qi and Kang, Fei and Jiang, Biao and An, Mengyin and Ren, Yangyang and Xu, Baixin and Guo, Hao-Xiang and Gong, Kaixiong and Wu, Cyrus and Li, Wei and Song, Xuchen and Liu, Yang and Li, Eric and Zhou, Yahui},
    journal={arXiv preprint arXiv:2508.13009},
    year={2025}
  }
```