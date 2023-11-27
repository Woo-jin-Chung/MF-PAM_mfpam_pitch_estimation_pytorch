# MF-PAM_pytorch

![overall_architecture](https://github.com/Woo-jin-Chung/mfpam-pitch-estimation-pytorch/assets/76720656/9771d5ca-9993-4e84-ae13-d6d7481abf0f)

This repo is the official Pytorch implementation of ["MF-PAM: Accurate Pitch Estimation through Periodicity Analysis and Multi-level Feature Fusion"](https://arxiv.org/abs/2306.09640) accepted in INTERSPEECH 2023.


In the paper we predicted the quantized f0 with BCELoss.

However, you can also directly estimate the f0 value with L1 loss, which gives a more accurate VAD performance.
You may use train_direct.py or model_direct.py for direct f0 estimation

## Dependencies
```
pip install -r requirements.txt
```

## Make json file for dataset
Prepare for clean dataset and the corresponding noisy dataset
```
python make_data_json.py path/to/clean/train/data/dir > json_save_dir/train/clean.json
python make_data_json.py path/to/noisy/train/data/dir > json_save_dir/train/noisy.json
python make_data_json.py path/to/clean/test/data/dir > json_save_dir/test/clean.json
python make_data_json.py path/to/noisy/test/data/dir > json_save_dir/test/noisy.json
```
Also prepare a room impulse response (RIR) wavfile list in line 134-137 dataset.py

## Training
```
python train.py --checkpoint_path /path/to/save/checkpoint --data_json_path /path/to/data/json/directory
```

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
