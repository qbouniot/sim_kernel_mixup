# Tailoring Mixup to Data for Calibration

Official code for Similarity Kernel Mixup for regression tasks.


## Prerequisites

You can install required packages through pip:
```bash
pip install -r requirements.txt
```

Or through conda:
```bash
conda env create -f environment.yml
```


## Datasets and Scripts

### Airfoil
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run Similarity Kernel Mixup on Airfoil is:
```
python main.py --dataset Airfoil --mixtype kernel_sim --mix_alpha 1. --tau_max_x 0.0001 --tau_max_y 0.0001 --tau_std_x 0.5 --tau_std_y 0.5 --use_dropout --mc_dropout --use_manifold 0 --store_model 0 --read_best_model 0 --seed 0
```

### Exchange_rate
This dataset can be downloaded via the link in the [Google Drive](https://drive.google.com/drive/folders/1pTRT7fA-hq6p1F7ZX5oJ0tg_I1RRG6OW?usp=sharing). Please put the corresponding datafolder to `src/data`


The command to run Similarity Kernel Mixup on Exchange_rate is:
```
python main.py --dataset TimeSeries --data_dir ./data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --mixtype kernel_sim --mix_alpha 1. --tau_max_x 5 --tau_max_y 5 --tau_std_x 1. --tau_std_y 1. --use_dropout --mc_dropout --use_manifold 0 --store_model 0 --read_best_model 0 --seed 0
```

## Acknowledgements

This code is based on the official code of the following paper:
```
@inproceedings{yao2022cmix,
  title={C-Mixup: Improving Generalization in Regression},
  author={Yao, Huaxiu and Wang, Yiping and Zhang, Linjun and Zou, James and Finn, Chelsea},
  booktitle={Proceeding of the Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022}
}
```