"""
Vendored from ICCV2019_MirrorNet (Yang, Mei, Xu, Wei, Yin, Lau, "Where Is My
Mirror?", ICCV 2019 - https://github.com/Mhaiyang/ICCV2019_MirrorNet).
See LICENSE.txt in this directory. Only the two `backbone.resnext...` import
lines in mirrornet.py / resnext101_regular.py were changed (to relative
imports so this vendors cleanly as a subpackage); the model code itself is
unmodified. Pretrained weights (MirrorNet.pth) are NOT vendored here - point
ANALYTICS_MIRRORNET_WEIGHTS at a separately downloaded checkpoint.
"""
