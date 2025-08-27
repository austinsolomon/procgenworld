<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

# Matrix-Game: Interactive World Foundation Model
<font size=7><div align='center' >  [[🤗 HuggingFace](https://huggingface.co/Skywork/Matrix-Game)] [[📖 Technical Report](https://arxiv.org/pdf/2506.18701)] [[🚀 Project Website](https://matrix-game-homepage.github.io/)] </div></font>

<div align="center">
  <img src="assets/videos/demo.gif" alt="teaser" />
</div>

## 📝 Overview
**Matrix-Game** is a 17B-parameter interactive world foundation model for controllable game world generation.

## ✨ Key Features

- 🎯 **Feature 1**: **Interactive Generation.**  A diffusion-based image-to-world model that generates high-quality videos conditioned on keyboard and mouse inputs, enabling fine-grained control and dynamic scene evolution.
- 🚀 **Feature 2**: **GameWorld Score.** A comprehensive benchmark for evaluating Minecraft world models across four key dimensions, including visual quality, temporal quality, action controllability, and physical rule understanding. 
- 💡 **Feature 3**: **Matrix-Game Dataset** A large-scale Minecraft dataset with fine-grained action annotations, supporting scalable training for interactive and physically grounded world modeling.

## 🔥 Latest Updates

* [2025-07] 🚀 Released parallel inference script
* [2025-05] 🎉 Initial release of Matrix-Game Model

## 🚀 Performance Comparison
### GameWorld Score Benchmark Comparison

| Model     | Image Quality | Aesthetic Quality | Temporal Cons. | Motion Smooth. | Keyboard Acc. | Mouse Acc. | Object Cons. | Scenario Cons.|
|-----------|------------------|-------------|-------------------|-------------------|------------------|---------------|-------------|-------------|
| Oasis     | 0.65             | 0.48        | 0.94              | **0.98**          | 0.77             | 0.56          | 0.56        |  0.86 | 
| MineWorld | 0.69             | 0.47        | 0.95              | **0.98**          | 0.86             | 0.64          | 0.51        | 0.92        |
| **Ours**  | **0.72**         | **0.49**    | **0.97**          | **0.98**          | **0.95**         | **0.95**      | **0.76**    | **0.93**    |

**Metric Descriptions**:

- **Image Quality** / **Aesthetic**: Visual fidelity and perceptual appeal of generated frames  
- **Temporal Consistency** / **Motion Smoothness**: Temporal coherence and smoothness between frames  
- **Keyboard Accuracy** / **Mouse Accuracy**: Accuracy in following user control signals  
- **Object Consistency**: Geometric stability and consistency of objects over time
- **Scenario Consistency**: Scenario consistency over time

  Please check our [GameWorld](https://github.com/SkyworkAI/Matrix-Game/tree/main/GameWorldScore) benchmark for detailed implementation.

### Human Evaluation

![Human Win Rate](assets/imgs/human_win_rate.png)

> Double-blind human evaluation by two independent groups across four key dimensions: **Overall Quality**, **Controllability**, **Visual Quality**, and **Temporal Consistency**.  
> Scores represent the percentage of pairwise comparisons in which each method was preferred. Matrix-Game consistently outperforms prior models across all metrics and both groups.


## 🚀 Quick Start

```
# clone the repository:
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game

# install dependencies:
pip install -r requirements.txt

# install apex and FlashAttention-3
# Our project also depends on [apex](https://github.com/NVIDIA/apex) and [FlashAttention-3](https://github.com/Dao-AILab/flash-attention)

# inference
bash run_inference.sh
```


## 🔧 Hardware Requirements
- **GPU**:
  - NVIDIA A100/H100
- **VRAM**:
  - Requires **≥80GB of GPU memory** for a single 65-frame video inference.


## ⭐ Acknowledgements

We would like to express our gratitude to:

- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) for their strong base model
- [MineDojo](https://minedojo.org/knowledge_base) for their Minecraft video dataset
- [MineRL](https://github.com/minerllabs/minerl) for their excellent gym framework
- [Video-Pre-Training](https://github.com/openai/Video-Pre-Training) for their accurate Inverse Dynamics Model
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module 

We are grateful to the broader research community for their open exploration and contributions to the field of interactive world generation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📎 Citation
If you find this project useful, please cite our paper:
```bibtex
@article{zhang2025matrixgame,
  title     = {Matrix-Game: Interactive World Foundation Model},
  author    = {Yifan Zhang and Chunli Peng and Boyang Wang and Puyi Wang and Qingcheng Zhu and Fei Kang and Biao Jiang and Zedong Gao and Eric Li and Yang Liu and Yahui Zhou},
  journal   = {arXiv preprint arXiv:2506.18701},
  year      = {2025}
}
```
