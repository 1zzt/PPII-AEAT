# PPII-AEAT: Pediction of Protein-Protein Interaction Inhibitors based on Autoencoders with Adversarial Training

## 1. overview
![image](https://github.com/1zzt/PPII-AEAT/raw/main/overview.png)
## 2. Prerequisites
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
## 3. Datasets
We evaluate the performance of PPII-AEAT on different tasks and PPI families. PPI families include **Bcl2-Like/Bak-Bax**, **Bromodomain/Histone**, **Cyclophilins**, **HIF-1a/p300**, **Integrins**, **LEDGF/IN**, **LFA/ICAM**, **Mdm2-Like/P53**, **Ras/SOS1**, and **XIAP/Smac**.

## 4. Usage
```
python main.py --dataset bcl2_bak --task regression --batch_size 32 --num_epochs 100 --lr 0.001 --gpu 0
```
