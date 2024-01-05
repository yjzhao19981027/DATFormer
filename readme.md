# DATFormer

Source code for our ACM MM23 paper "Distortion-aware Transformer in 360° Salient Object Detection" by Yinjie Zhao, Lichen Zhao, Qian Yu, Jing Zhang, Lu Sheng, Dong Xu

![](https://github.com/yjzhao19981027/DATFormer/blob/main/pipeline.jpg)

## Requirements
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
pip install opencv-python
pip install scipy
pip install matplotlib
pip install timm
```

## Data Preparation


Your /DATFormer/data folder should look like this:
```
-- data
   |-- 360-SOD
   |   |-- train
   |   |-- | images
   |   |-- | labels
   |   |-- | contours
   |   |-- test
   |   |-- | images
   |   |-- | labels
   |-- F-360iSOD
   |   |-- train
   |   |-- | images
   |   |-- | labels
   |   |-- | contours
   |   |-- test
   |   |-- | images
   |   |-- | labels
   |-- 360-SSOD
   |   |-- train
   |   |-- | images
   |   |-- | labels
   |   |-- | contours
   |   |-- test
   |   |-- | images
   |   |-- | labels
```

## Training, Testing, and Evaluation
- ```cd DATFormer```
- Download the pretrained [T2T-ViT_t-14v2]( https://drive.google.com/file/d/1HLQjWE6x7dp1b3bJXOVJu0hRDFvcZbXD/view?usp=sharing) model  and put it into ```pretrained_model/``` folder.
- Run ```python train_test_eval.py --Training True --Testing True --Evaluation True``` for training, testing, and evaluation. The predictions will be in preds/ folder and the evaluation results will be in result.txt file.

## Acknowledgement

Thanks to the author of VST for providing the VST model and related codes for training and testing.

## Citation

```
@inproceedings{zhao2023distortion,
  title={Distortion-aware Transformer in 360° Salient Object Detection},
  author={Zhao, Yinjie and Zhao, Lichen and Yu, Qian and Sheng, Lu and Zhang, Jing and Xu, Dong},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={499--508},
  year={2023}
}
```