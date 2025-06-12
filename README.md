# SACB-Net: Spatial-awareness Convolutions for Medical Image Registration
The official implementation of SACB-Net [![CVPR](https://img.shields.io/badge/CVPR2025-68BC71.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_SACB-Net_Spatial-awareness_Convolutions_for_Medical_Image_Registration_CVPR_2025_paper.html)  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.19592) 

## Env
```
#pip < 24.1
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```
## Dataset
Thanks [@Junyu](https://github.com/junyuchen245) for [the preprocessed IXI data]
[Abdomen CT-CT](https://learn2reg.grand-challenge.org/Datasets/)
[LPBA](https://loni.usc.edu/research/atlases)

## Weights Download
[Google Drive](https://drive.google.com/drive/folders/1XW19iuyCyg3YGmCpLFGGFjdPFi73xxwh?usp=share_link).

## Citation
```bibtex
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Xinxing and Zhang, Tianyang and Lu, Wenqi and Meng, Qingjie and Frangi, Alejandro F. and Duan, Jinming},
    title     = {SACB-Net: Spatial-awareness Convolutions for Medical Image Registration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5227-5237}
}
```

## Acknowledgments
We sincerely acknowledge the [ModeT](https://github.com/ZAX130/SmileCode), [CANNet](https://github.com/Duanyll/CANConv) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects.