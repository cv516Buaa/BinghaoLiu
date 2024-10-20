
<p align="center">
  <h1 align="center">African water body segmentation with cross-layer information separability based feature decoupling transformer</h1>
  <p align="center">


   <br />
    <strong>Binghao Liu</strong></a>
    ·
    <strong>Qi Zhao</strong></a>
    ·
    <strong>Chunlei Wang</strong></a>
    ·
    <strong>Meng Li</strong></a>    
    ·
    <strong>Hongbo Xie</strong></a>    
    ·
    <strong>Lijiang Chen</strong></a>
    <br />
<p align="center">
 </p>





## Introduction
This repo is the implementation of "African water body segmentation with cross-layer information separability based feature decoupling transformer"

The data distribution and several examples of our AWS16K dataset.

<p align="center">
  <img src="images/AWS16K.png" width="720">
</p>

AWS16K contains 16888 water body images covering the whole African area.

The cross-layer information separability based Feature Decoupling Transformer:

<p align="center">
  <img src="images/FDTran.png" width="720">
</p>

## Usage

### Install

Clone [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) repo and add the codes of `configs`, `mmseg/datasets/aws.py` and `mmseg/models/decode_heads/fdt_head.py` into corresding files of MMSegmentation.

Then, run
`pip install -v -e .`
to regist AWS16K dataset and FDTran model.

### Dataset

Our AWS16K dataset can be found at [AWS16K](https://pan.baidu.com/s/1bUi_k14JIQt2Wbemx6LLTg) (password: awbs), then you need to convert 255 pixels to 1 for training and testing.

### Train and Test

+ Use the following command for training

  ```
  python tools/train.py \
  config_path \
  --work-dir work_path
  ```

+ Use the following command for testing with TResNet-L

  ```
  python tools/test.py \
  config_path \
  ckpt_path \
  --work-dir work_path
  ```

The pretrained weights of FDTran (with mit_b5 as backbone) can be found at [https://pan.baidu.com/s/1Gbh0SZiswnzykhwb5z_PcQ](https://pan.baidu.com/s/1Gbh0SZiswnzykhwb5z_PcQ) (password: fdt1).

## Citation

If you have any question, please discuss with me by sending email to liubinghao@buaa.edu.cn

## References

The code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks for their great works!
