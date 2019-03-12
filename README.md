# SwiftNet

Source code to reproduce results from 
<div class="highlight highlight-html"><pre>
<b>In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images
<a href=https://github.com/orsic>Marin Oršić*</a>, <a href=https://ivankreso.github.io/>Ivan Krešo*</a>, <a href=http://www.zemris.fer.hr/~ssegvic/index_en.html>Siniša Šegvić</a>, Petra Bevandić (* denotes equal contribution)</b>
CVPR, 2019.
</pre></div>

## Steps to reproduce

### Install requirements
* Python 3.7+ 
```bash
pip install -r requirements.txt
```

### Download pre-trained models
```bash
wget http://elbereth.zemris.fer.hr/swiftnet/swiftnet_ss_cs.pt -P weights/
wget http://elbereth.zemris.fer.hr/swiftnet/swiftnet_pyr_cs.pt -P weights/
```

### Download Cityscapes

From https://www.cityscapes-dataset.com/downloads/ download: 
* leftImg8bit_trainvaltest.zip (11GB)
* gtFine_trainvaltest.zip (241MB)

Either download and extract to `datasets/` or create a symbolic link `datasets/Cityscapes`

### Evaluate
```bash
python eval.py configs/single_scale.py
python eval.py configs/pyramid.py
``` 