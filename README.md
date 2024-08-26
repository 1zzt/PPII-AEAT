# PPII-AEAT: Pediction of Protein-Protein Interaction Inhibitors based on Autoencoders with Adversarial Training

## Overview
![image](https://github.com/1zzt/PPII-AEAT/raw/main/overview.jpg)
PPII-AEAT is a new PPI inhibitor prediction method based on an autoencoder with adversarial training that adaptively learns molecular representations to cope with different PPI targets. The three parts of PPII-AEAT are primary feature encoding, feature representation learning, and inhibitory score prediction. In the first part, small molecular compounds are encoded into primary features based on Extended-connectivity fingerprints and Mordred descriptors. In the second part, the primary features are fed to an autoencoder-based deep neural network trained with three phases to generate high-level feature representations. In the third part, the learned high-level representations are used to predict inhibitory scores. For the classification task, the threshold of the prediction scores is 0.5; that is, samples with scores greater than 0.5 are identified as PPI inhibitors. For the regression task, the predicted scores are regarded as inhibitory potency.

## Prerequisites
- python 3.9.13
- pytorch 1.11.0
- numpy 1.22.3
- pandas 1.4.3
- mordred 1.2.0
- rdkit 2022.03.3
- scikit-learn 1.0.2
- scipy 1.9.0
- tqdm 4.64.0
- prettytable 3.3.0

## Datasets
We evaluate the performance of PPII-AEAT on different tasks and PPI families. 
PPI families include **Bcl2-Like/Bak-Bax**, **Bromodomain/Histone**, **Cyclophilins**, **HIF-1a/p300**, **Integrins**, **LEDGF/IN**, **LFA/ICAM**, **Mdm2-Like/P53**, **Ras/SOS1**, and **XIAP/Smac**.

## Usage
```
python main.py --dataset bcl2_bak --task regression --batch_size 32 --num_epochs 100 --lr 0.001 --gpu 0 --seed 24

Parameters:
--dataset <PPI name>             bcl2_bak; bromodomain_histone; cyclophilins; hif1a_p300; integrins; ledgf_in; lfa_icam; mdm2_p53; xiap_smac; default=bcl2_bak.
--task <Prediction task>         classification or regression; default=regression.
--batch_size <Batch size>        default=32.
--num_epochs <Training epochs>   default=100.
--gpu <Number of gpu>            default=0.
--seed <Random seed>             default=24.
```
The results are presented in the Prettytable.

## Cite us

```
@article{zhang2024ppii,
  title={PPII-AEAT: Prediction of protein-protein interaction inhibitors based on autoencoders with adversarial training},
  author={Zhang, Zitong and Zhao, Lingling and Gao, Mengyao and Chen, Yuanlong and Wang, Junjie and Wang, Chunyu},
  journal={Computers in Biology and Medicine},
  volume={172},
  pages={108287},
  year={2024},
  publisher={Elsevier}
}
```
