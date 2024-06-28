# ruff: noqa: F401
from .batch import MIMOBatchFormat, RepeatTarget
from .cutout import Cutout
from .image import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    Posterize,
    RandomRescale,
    Rotate,
    Sharpen,
    Shear,
    Solarize,
    Translate,
)
from .mixup import (
    MITMixup,
    Mixup,
    MixupIO,
    MixupTO,
    QuantileMixup,
    RankMixupMNDCG,
    RegMixup,
    KernelMixup,
)

augmentations = [
    AutoContrast,
    Equalize,
    Posterize,
    Rotate,
    Solarize,
    Shear,
    Translate,
    Contrast,
    Brightness,
    Color,
    Sharpen,
]
