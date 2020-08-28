# SwiftNet

Source code to reproduce results from 
<div class="highlight highlight-html"><pre>
<b><a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320320304143">Efficient semantic segmentation with pyramidal fusion</a></b>
<a href=https://github.com/orsic>Marin Oršić</a>, <a href=http://www.zemris.fer.hr/~ssegvic/index_en.html>Siniša Šegvić</a>
Pattern Recognition, 2020.
</pre>

<pre>
<b><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Orsic_In_Defense_of_Pre-Trained_ImageNet_Architectures_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.pdf">In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images</a>
<a href=https://github.com/orsic>Marin Oršić*</a>, <a href=https://ivankreso.github.io/>Ivan Krešo*</a>, <a href=http://www.zemris.fer.hr/~ssegvic/index_en.html>Siniša Šegvić</a>, Petra Bevandić (* denotes equal contribution)</b>
CVPR, 2019.
</pre>

</div>

## Steps to reproduce

### Install requirements
* Python 3.7+ 
```bash
pip install -r requirements.txt
```

### Download Cityscapes

From https://www.cityscapes-dataset.com/downloads/ download: 
* leftImg8bit_trainvaltest.zip (11GB)
* gtFine_trainvaltest.zip (241MB)

Either download and extract to `datasets/` or create a symbolic link `datasets/Cityscapes`
Expected dataset structure for Cityscapes is:
```
labels/
    train/
        aachen/
            aachen_000000_000019.png
            ...
        ...
    val/
        ...
rgb/
    train/
        aachen/
            aachen_000000_000019.png
            ...
        ...
    val/
        ...
```


### Evaluate
##### Pre-trained Cityscapes models [available](https://drive.google.com/drive/folders/1DqX-N-nMtGG9QfMY_cKtULCKTfEuV4WT?usp=sharing)
* Download and extract to `weights` directory.

Set `evaluating = True` inside config file (eg. `configs/rn18_single_scale.py`) and run:
```bash
python eval.py configs/rn18_single_scale.py
``` 

### Train
```bash
python train.py configs/rn18_single_scale.py --store_dir=/path/to/store/experiments
``` 
