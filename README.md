<div align="center">

<h1>Realistic Full-Body Tracking from Sparse Observations via Joint-Level Modeling</h1>

<div>
    <a href='https://scholar.google.com/citations?user=3hSD41oAAAAJ' target='_blank'>Xiaozheng Zheng<sup>†</sup></a>&emsp;
    <a href='https://suzhuo.github.io/' target='_blank'>Zhuo Su<sup>†</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=v8TFZI4AAAAJ' target='_blank'>Chao Wen</a>&emsp;
    <a href='https://scholar.google.com/citations?&user=ECKq3aUAAAAJ' target='_blank'>Zhou Xue<sup>*</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=OEZ816YAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Xiaojie Jin</a>&emsp;
</div>
<div>
    ByteDance
</div>

<div>
    <sup>†</sup>Equal contribution &emsp; <sup>*</sup>Corresponding author
</div>

<div>
    :star_struck: <strong>Accepted to ICCV 2023</strong>
</div>

---

<strong> AvatarJLM uses tracking signals of the head and hands to estimate accurate, smooth, and plausible full-body motions. </strong>
<img src="assets/teaser.jpg" width="43.9%"/><img src="assets/2-1-crop.gif" width="56.1%"/>

:open_book: For more visual results, go checkout our <a href="https://zxz267.github.io/AvatarJLM/" target="_blank">project page</a>

---

<h4 align="center">
  <a href="https://zxz267.github.io/AvatarJLM/" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/abs/2308.08855" target='_blank'>[arXiv]</a>
</h4>

</div>

## :mega: Updates
[09/2023] Testing samples are available.

[09/2023] Training and testing codes are released.

[07/2023] AvatarJLM is accepted to ICCV 2023:partying_face:!


## :file_folder: Data Preparation
### AMASS
1. Please download the datasets from [AMASS](https://amass.is.tue.mpg.de/).
2. Download the required body models and placed them in `./support_data/body_models` directory of this repository. For the SMPL+H body model, download it from http://mano.is.tue.mpg.de/. Please download the AMASS version of the model with DMPL blendshapes. You can obtain dynamic shape blendshapes, e.g. DMPLs, from http://smpl.is.tue.mpg.de.
3. Run  `./data/prepare_data.py` to preprocess the input data for faster training. The data split for training and testing data under Protocol 1 in our paper is stored under the folder `./data/data_split` (from [AvatarPoser](https://github.com/eth-siplab/AvatarPoser)).
```
python ./data/prepare_data.py --protocol [1, 2, 3] --root [path to AMASS]
```
### Real-Captured Data
1. Please download our real-captured testing data from [Google Drive](https://drive.google.com/file/d/100v17S8a6w13t6TEo_L9mBgLLXMPfX0a/view?usp=sharing). The data is preprocessed to the same format as our preprocessed AMASS data.
2. Unzip the data and place it in `./data` directory of this repository.

## :desktop_computer: Requirements
### 
- Python >= 3.9
- PyTorch >= 1.11.0
- pyrender
- trimesh
- [human_body_prior](https://github.com/nghorbani/human_body_prior)
- [body_visualizer](https://github.com/nghorbani/body_visualizer)
- [mesh](https://github.com/MPI-IS/mesh)



## :bicyclist: Training
```
python train.py --protocol [1, 2, 3] --task [name of the experiment] 
```


## :running_woman: Evaluation
```
python test.py --protocol [1, 2, 3, real] --task [name of the experiment] --checkpoint [path to trained checkpoint] [--vis]
```

## :lollipop: Trained Model
| Protocol   | MPJRE  | MPJPE  | MPJVE  | Trained Model |
| :--------- | :----: | :----: | :----: |:-------------:|
| 1          | 3.01   | 3.35   | 21.01  |[Google Drive](https://drive.google.com/file/d/1yXM8VT04L9zAHUpjEn3iV7Sck1U9LkNj/view?usp=sharing) |
| 2-CMU-Test | 5.36   | 7.28   | 26.46  |[Google Drive](https://drive.google.com/file/d/1aEKPD5IR38e7iKMpokBSrJBk2hKzYRfx/view?usp=drive_link) |
| 2-BML-Test | 4.65   | 6.22   | 34.45  |[Google Drive](https://drive.google.com/file/d/1k1Um73tG3F0Bv5B-2MRvEHAIpSExdWt_/view?usp=sharing) |
| 2-MPI-Test | 5.85   | 6.47   | 24.13  |[Google Drive](https://drive.google.com/file/d/1LeAEHsTbAjutqPljqQ1TQNH2hCaep6j3/view?usp=sharing) |
| 3          | 4.25   | 4.92   | 27.04  |[Google Drive](https://drive.google.com/file/d/17O-6FQCFAC2ZJiP1HSQcBw8q6Msbc85L/view?usp=drive_link) |

## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{
  zheng2023realistic,
  title={Realistic Full-Body Tracking from Sparse Observations via Joint-Level Modeling},
  author={Zheng, Xiaozheng and Zhuo Su and Wen, Chao and Xue, Zhou and Xiaojie Jin},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  year={2023}
}
```

## :newspaper_roll: License

Distributed under the MIT License. See `LICENSE` for more information.

## :raised_hands: Acknowledgements
This project is built on source codes shared by [AvatarPoser](https://github.com/eth-siplab/AvatarPoser). We thank the authors for their great job!
