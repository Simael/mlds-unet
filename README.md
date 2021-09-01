# Multi-label Deeply Supervised UNet

## Overview
This repository contains a pytorch implementation of the network architecture and loss from the paper ["Every annotation counts: Multi-label deep supervision for medical image segmentation"](https://openaccess.thecvf.com/content/CVPR2021/html/Reiss_Every_Annotation_Counts_Multi-Label_Deep_Supervision_for_Medical_Image_Segmentation_CVPR_2021_paper.html).

## UNet architecture
The pytorch implementation for a standard [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) was extended from a [previous implementation](https://github.com/milesial/Pytorch-UNet) and can be found in [unet.py](https://github.com/Simael/mlds-unet/blob/master/unet.py).
The class "HierarchicalUNet" is an UNet architecture extended with additional output-heads in the decoder, in the paper the output is (1) passed to the multi-label deep supervision function and (2) the full-scale output is passed to a standard cross-entropy loss function as well.

## Multi-label Deep Supervision loss-function
In the file [loss_function.py](https://github.com/Simael/mlds-unet/blob/master/loss_function.py) the multi-label deep supervision loss can be found.
The input to this function is a list of logit outputs from the intermediate output-heads and the full-scale target.
As described in the paper, the target is downscaled and leveraged for the intermediate outputs.

## Cite
```latex
@inproceedings{reiss2021every,
  title={Every annotation counts: Multi-label deep supervision for medical image segmentation},
  author={Rei{\ss}, Simon and Seibold, Constantin and Freytag, Alexander and Rodner, Erik and Stiefelhagen, Rainer},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9532--9542},
  year={2021}
}

