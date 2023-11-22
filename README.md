# MF-PAM_pytorch

![overall_architecture](https://github.com/Woo-jin-Chung/mfpam-pitch-estimation-pytorch/assets/76720656/9771d5ca-9993-4e84-ae13-d6d7481abf0f)

This repo is the official Pytorch implementation of ["MF-PAM: Accurate Pitch Estimation through Periodicity Analysis and Multi-level Feature Fusion"](https://arxiv.org/abs/2306.09640) accepted in INTERSPEECH 2023.


In the paper we predicted the quantized f0 with BCELoss.

However, you can also directly estimate the f0 value with L1 loss, which gives a more accurate VAD performance. (To Do)


## Publications
```
@inproceedings{chung23_interspeech,
  author={Woo-Jin Chung and Doyeon Kim and Soo-Whan Chung and Hong-Goo Kang},
  title={{MF-PAM: Accurate Pitch Estimation through Periodicity Analysis and Multi-level Feature Fusion}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={4499--4503},
  doi={10.21437/Interspeech.2023-2487}
}
```
