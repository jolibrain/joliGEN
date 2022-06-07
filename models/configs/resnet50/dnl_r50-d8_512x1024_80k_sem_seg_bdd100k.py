"""DNL with ResNet-50-d8."""

_base_ = [
    "./dnl_r50-d8.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/dnl_r50-d8_512x1024_80k_sem_seg_bdd100k.pth"
