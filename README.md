##  SLOTMFound: Foundation-Based Diagnosis of Multiple Sclerosis Using Retinal SLO Imaging and OCT Thickness-maps.



Official repo for [SLOTMFound: Foundation-Based Diagnosis of Multiple Sclerosis Using Retinal SLO Imaging and OCT Thickness-maps], which is based on [MAE](https://github.com/facebookresearch/mae):

Please contact 	**reyhaneh.esmailizadeh.ut@gmail.com** if you have questions.
You can find the paper here: https://www.medrxiv.org/content/10.1101/2025.07.14.25331522v1

### Key features

- TMFound is the RetFound-Fundus, which is fine-tuned on 1469 thickness maps with self-supervised learning
- TMFound has been validated in MS disease detection tasks
- TMFound can be efficiently adapted to customised tasks

- SLOFound is the RetFound-Fundus, which is fine-tuned on 203 SLO images with self-supervised learning
- SLOFound has been validated in MS disease detection tasks
- SLOFound can be efficiently adapted to customised tasks
  
Before heading into next section, make sure you have this repository in your github:
https://github.com/Reyhaneesmailizadeh/RETFound-Edited

### Install environment

1. Create environment with conda:

```
conda create -n retfound python=3.7.5 -y
conda activate retfound
```

2. Install dependencies

```
git clone https://github.com/Reyhaneesmailizadeh/RETFound-Edited
cd RETFound-Edited
pip install -r requirement.txt
```


### 📌 Fine-tuning with TMFound/SLOFound weights. Self-supervised approach

To fine tune TMFound/SLOFound on your own unlabeled data with self-supervised learning approach, follow these steps:

1. Download the TMFound/SLOFound pre-trained weights or add them to your drive:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">TMFound</td>
<td align="center"><a href="https://drive.google.com/file/d/1sk1IAdBaQ60qTCGqOQw5_s6E1tQj4Ftk/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">SLOFound</td>
<td align="center"><a href="https://drive.google.com/file/d/1-MCvsW4NWEAMoXNe5VAJrGat6z6GlUkV/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

2. Organise your data into this directory structure
   
Both TMFound and SLOFound expects the input image to be in this shape: (224,224,3). Hence, make sure the number of channel of your image data is 3.
```
├── data folder
    ├──train
        ├──folder_a
        ├──folder_b
        ├──folder_c
    ├──valid
        ├──folder_a
        ├──folder_b
        ├──folder_c
``` 
Example of our data directory for fine-tuning RetFound-Fundus on thickness maps is as follow. TM is an abbreviation of thickness map.
```
├── data folder
    ├──train
        ├──GCIPL_TM
            |1.png
            |2.png
            |3.png
        ├──RNFL_TM
            |1.png
            |2.png
            |3.png
        ├──TotalRetina_TM
            |1.png
            |2.png
            |3.png
    ├──valid
        ├──GCIPL_TM
            |1.png
            |2.png
            |3.png
        ├──RNFL_TM
            |1.png
            |2.png
            |3.png
        ├──TotalRetina_TM
            |1.png
            |2.png
            |3.png
```
3. Start fine-tuning. A fine-tuned checkpoint will be saved during training.

Fine-tuning TMFound on your own unlabeled data:
```
#Code from https://github.com/rmaphoh/RETFound_MAE with minor changes
%%shell
cd RETFound-Edited
eval "$(conda shell.bash hook)" # copy conda command to shell
source activate retfound
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_pretrain.py \
    --batch_size 4 \
    --world_size 1 \
    --epochs 50 \
    --blr 5e-3 \
    --weight_decay 0.05 \
    --input_size 224 \
    --output_dir /content/drive/MyDrive/checkpoints/ \
    --finetune /content/drive/MyDrive/ThickFound.pth \
    --log_dir /content/drive/MyDrive/logs/ \
    --data_path /content/drive/MyDrive/path-to-your-data/

```
Fine-tuning SLOFound on your own unlabeled data:
```
#Code from https://github.com/rmaphoh/RETFound_MAE with minor changes
%%shell
cd RETFound-Edited
eval "$(conda shell.bash hook)" # copy conda command to shell
source activate retfound
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_pretrain.py \
    --batch_size 4 \
    --world_size 1 \
    --epochs 50 \
    --blr 5e-3 \
    --weight_decay 0.05 \
    --input_size 224 \
    --output_dir /content/drive/MyDrive/checkpoints/ \
    --finetune /content/drive/MyDrive/SLOFound.pth \
    --log_dir /content/drive/MyDrive/logs/ \
    --data_path /content/drive/MyDrive/path-to-your-data/
```



Requirements for running the code of Best-MultiModal-Model.ipynb is as follow:
```
Package                  Version
------------------------ ----------
brotlipy                 0.7.0
certifi                  2021.10.8
cffi                     1.15.0
charset-normalizer       2.0.4
conda                    4.11.0
conda-content-trust      0+unknown
conda-package-handling   1.7.3
cryptography             36.0.0
filelock                 3.12.2
fsspec                   2023.1.0
huggingface-hub          0.16.4
idna                     3.3
importlib-metadata       6.7.0
numpy                    1.21.6
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
packaging                24.0
Pillow                   9.5.0
pip                      21.2.2
pycosat                  0.6.3
pycparser                2.21
pyOpenSSL                21.0.0
PySocks                  1.7.1
PyYAML                   6.0.1
requests                 2.27.1
ruamel-yaml-conda        0.15.100
safetensors              0.4.5
setuptools               58.0.4
six                      1.16.0
timm                     0.9.12
torch                    1.13.1
torchvision              0.14.1
tqdm                     4.62.3
typing_extensions        4.7.1
urllib3                  1.26.7
wheel                    0.37.1
zipp                     3.15.0
```

## Citing our paper
please include this in your work if you want to use any part of the work here:
```
This work has been accepted to ICMV 2025. A preprint is available at medRxiv.
Please cite the preprint until the official conference version is published.
```
