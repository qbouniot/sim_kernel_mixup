from setuptools import setup

setup(
   name='torch_uncertainty',
   version='0.2.0',
   packages=["torch_uncertainty"],
   install_requires=[
    "timm",
    "lightning[pytorch-extra]",
    "torchvision>=0.16",
    "tensorboard",
    "einops",
    "torchinfo",
    "scipy",
    "huggingface-hub",
    "scikit-learn",
    "matplotlib==3.5.2",
    "numpy<2",
    "opencv-python",
    "glest==0.0.1a0",
],
)