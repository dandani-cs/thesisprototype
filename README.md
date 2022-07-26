# Capsule Networks Classification of Mask and Usage

This repository contains prototype for the paper Enhancing CNN-based Face Mask Detection and Usage Classification using Capsule Networks.

Capsule Network Code [reference](https://google.com).

The model has 4 classes
1. Correctly Worn (face_with_mask)
1. Incorrectly Worn (face_with_mask_incorrect)
1. No Mask (face_no_mask)
1. Invalid Mask (face_other_covering)


The model has several issues:
- dark colored masks returns invalid mask
- cannot accurately classify with low resolutions


## Requirements:
1. Tkinter
1. Pillow

```python
pip install tkinter, pillow
```

The model weights for the YOLOv3 Face Detector and CapsNet Mask Classifier are in this [drive link](https://drive.google.com/drive/folders/18JsqyM3uuOPVExl3nN0tHUM0iu1MBFCX?usp=sharing).

Download the weights and place them inside the models folder inside the repository folder. 
