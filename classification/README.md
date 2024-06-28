# Image Classification experiments

This folder contains code to reproduce classification experiments from [Tailoring Mixup to Data for Calibration](https://arxiv.org/pdf/2311.01434).

The code presented here is based on the TorchUncertainty library (v0.2.0), shared here with the exact config files used for reproducibility of the experiments. We refer the interested reader to [the official github of TorchUncertainty](https://github.com/ENSTA-U2IS-AI/torch-uncertainty) for a more up-to-date version of the library.


## Installation

TorchUncertainty requires Python 3.10 or greater. Install the desired PyTorch version in your environment.
Then, install *this local version of TorchUncertainty* with:

```sh
pip install -e .
```

## Mixup Baselines

To date, the following baseline Mixup methods have been implemented:

- Mixup [ICLR 2021](http://arxiv.org/abs/1710.09412)
- MixupIO [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf)
- MixupTO [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf)
- RegMixup [NeurIPS 2022](https://arxiv.org/abs/2206.14502)
- MIT-Mixup [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf)
- RankMixup [ICCV 2023](https://arxiv.org/pdf/2308.11990)

## Running an experiment

To train a model on, e.g. CIFAR10 with a Resnet34 using *Similarity Kernel Mixup*, use the following command:

```sh
python3 ./experiments/classification/cifar10/resnet.py fit --config ./experiments/classification/cifar10/configs/resnet34/sk_mixup.yaml
```

Default parameters can be changed in the YAML files, or directly from the command line. Config files for other methods, architectures and datasets can be found in `experiments/classification`.

Evaluation is done automatically at the end of training. However, use the following the command to evaluate a specific checkpoint:

```sh
python3 ./experiments/classification/cifar10/resnet.py test --config ./experiments/classification/cifar10/configs/resnet34/sk_mixup.yaml --ckpt_path="path/to/checkpoint.ckpt"
```

## More about TorchUncertainty

<div align="center">

![TorchUncertaintyLogo](https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/docs/source/_static/images/torch_uncertainty.png)

</div>

_TorchUncertainty_ is a package designed to help you leverage [uncertainty quantification techniques](https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning) and make your deep neural networks more reliable. It aims at being collaborative and including as many methods as possible, so reach out to add yours!

Our webpage and documentation is available here: [torch-uncertainty.github.io](https://torch-uncertainty.github.io).
