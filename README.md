##  MS-Diagnosis-Using-Novel-Foundation-Multimodal-and-SLO-images


Official repo for [MS Diagnosis Using Novel Foundation Multimodal and-SLO images], which is based on [MAE](https://github.com/facebookresearch/mae):

Please contact 	**reyhaneh.esmailizadeh.ut@gmail.com** if you have questions.


### Key features

- ThickFound is the RetFound-Fundus which is pre-trained on 1469 thickness maps with self-supervised learning
- ThickFound has been validated in multiple disease detection tasks
- ThickFound can be efficiently adapted to customised tasks

- SLOFound is the RetFound-Fundus which is pre-trained on 203 SLO images with self-supervised learning
- SLOFound has been validated in multiple disease detection tasks
- SLOFound can be efficiently adapted to customised tasks


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


### 🌱Fine-tuning with ThickFound weights

To fine tune ThickFound/SLOFound on your own data, follow these steps:

1. Download the ThickFound/SLOFound pre-trained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">ThickFound</td>
<td align="center"><a href="https://drive.google.com/file/d/1sk1IAdBaQ60qTCGqOQw5_s6E1tQj4Ftk/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">SLOFound</td>
<td align="center"><a href="https://drive.google.com/file/d/1-MCvsW4NWEAMoXNe5VAJrGat6z6GlUkV/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

2. Organise your data into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──valid
        ├──class_a
        ├──class_b
        ├──class_c
``` 

3. Start fine-tuning. A fine-tuned checkpoint will be saved during training. Evaluation will be run after training.


```
#Code from https://github.com/rmaphoh/RETFound_MAE
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

