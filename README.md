# IRMCon-IPW
This repository is the PyTorch 
implementation for ECCV 2022 Paper 
"Class Is Invariant to Context and Vice Versa: On Learning Invariance for Out-Of-Distribution Generalization", 
For the detailed theories, 
please refer to our paper. If you have any questions or suggestions, 
please email me, (I do not usually browse my 
Github, so the reply to issues may be not on time).

Note that this repository is based on the [LfF](https://github.com/alinlab/LfF) and [DomainBed](https://github.com/facebookresearch/DomainBed). (Note, the data processing in Lff should be checked when you use their code, the input range seems abnormal.)

If you find this work is useful in your research, please kindly consider citing:
```
@inproceedings{qi2022class,
  title={Class Is Invariant to Context and Vice Versa: On Learning Invariance for Out-Of-Distribution Generalization},
  author={Qi, Jiaxin and Tang, Kaihua and Sun, Qianru and Hua, Xian-Sheng and Zhang, Hanwang},
  booktitle={ECCV},
  year={2022}
}
```
##### Dependencies
python 3.9.4, pytorch 1.7.1, torchvision 0.8.2 
##### Preparing
Download biased data from [here](https://drive.google.com/file/d/1hJlYVDDjr_dBZ3HMJgrpVF9VmvNXVWM5/view?usp=sharing) and unzip it under the path ./Biased_dataset/data (Note the result should be ""./Biased_dataset/data/cmnist/...")
##### Training (examples)

1.Biased dataset, Check your download data path and the set data path in the code.

1.1.Training for Colored MNIST

ERM baseline: python train_cmnist_erm.py --dir_name bias0.05

Ours: python train_cmnist_ours.py --dir_name bias0.05

1.2.Training for Corrupted Cifar-10

ERM baseline: python train_ccifar10_erm.py --dir_name bias0.05

Ours: python train_ccifar10_ours.py --dir_name bias0.05

1.3.Training for BAR

ERM baseline: python train_bar_erm.py --ratio 0.05

Ours: python train_bar_ours.py --ratio 0.05

2.Training for PACS (codebase is from DomainBed, find the full version from [here](https://github.com/facebookresearch/DomainBed))

2.1.download data(PACS) from DomainBed and put into ./data

2.2.run baseline: python train_ERM.py  --add_note pacs_erm --test_envs 0 --no_pretrain --use_res18 --epochs 100

2.3.run ours: python train_ours.py  --add_note pacs_ours --test_envs 0 --no_pretrain --use_res18 --epochs 100

##### Acknowledgements

Thanks for the source code from LfF and DomainBed.
