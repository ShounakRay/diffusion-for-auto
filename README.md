# Forked - AV Applications: Latent Diffusion Models
[arXiv](https://arxiv.org/abs/2112.10752) | [BibTeX](#bibtex)

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

<p align="center">
<img src=assets/modelfigure.png />
</p>

## Requirements

### Initialize Conda Environment
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```shell script
conda env create -f environment.yaml
conda activate ldm-shounak
```
#### Fix #1
Now, inside this conda environment, run:
```shell script
python3 -m pip install packaging==21.3
```
This is to avoid an error involving PyTorch Lighting.
This is the error (if you don't have packaging==21.3 on your system):
```
packaging.version.InvalidVersion: Invalid version: '0.10.1,<0.11'
```

#### Fix #2
Also, inside this conda environment, run:
```shell script
python3 -m pip install torchmetrics==0.6
```
This is to avoid an error involving PyTorch Lighting.
This is the error (if you don't have torchmetrics==0.6 on your system):
```
ImportError: cannot import name 'get_num_classes' from 'torchmetrics.utilities.data'
```

For more information, refer to: <https://github.com/CompVis/latent-diffusion/issues/207#issuecomment-1377329827> as required.

### Retrieve a Pretrained VAE (for now)
You also need to install the model checkpoint for the pre-trained autoencoder we're going to use.
From the root directory of this repository:

```shell script
wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
cd models/first_stage_models/vq-f4
unzip -o model.zip
```

### Confirm LDM Training Dataset Paths
#### References to train and validation sets
Lastly, ensure that the driving data is in the right location. We will use the waymo perception dataset as an example. Specifically, the following two files should exist (with respect to the root directory):
```
data/autodrive/waymo_train.txt
data/autodrive/waymo_val.txt
```
Each text file is a list of all the image-paths that belong to the training and validation sets. For example, the first few lines of ``data/autodrive/waymo_train.txt`` looks like:
```
00153297.jpg
00075449.jpg
00120038.jpg
00062245.jpg
00055714.jpg
...
```
Notice that there aren't any other prefixes to each file path. Ensure that you don't add a prefix here. See the docstrings above ``AUTOWaymoTrain`` and ``AUTOWaymoValidation`` located inside ``ldm/data/autodrive.py`` for how these text files were created.

#### References to the original data
Furthermore, the driving dataset should exist as well in a particular location as defined in ``AUTOWaymoTrain`` and ``AUTOWaymoValidation`` located inside ``ldm/data/autodrive.py``.
The current, hardcoded ``data_root`` is 
```/scratch/groups/mykel/shounak_files/DATASETS/waymo/all```.

---
---

# Training autoencoder models
## Using a Pretrained Version
We've already downloaded a pretrained AE model (SEE ""Retrieve a Pretrained VAE (for now)"" above).
Visit the original repository for more information

## Doing it from scratch
Configs for training a KL-regularized autoencoder on ImageNet are provided at `configs/autoencoder`.
Training can be started by running
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,    
```
where `config_spec` is one of {`autoencoder_kl_8x8x64`(f=32, d=64), `autoencoder_kl_16x16x16`(f=16, d=16), 
`autoencoder_kl_32x32x4`(f=8, d=4), `autoencoder_kl_64x64x3`(f=4, d=3)}.

For training VQ-regularized models, see the [taming-transformers](https://github.com/CompVis/taming-transformers) 
repository.

# Training LDMs
## Using a Pretrained AV version
Stay tuned.

## Doing it from scratch
In ``configs/latent-diffusion/`` we provide configs for training LDMs on the LSUN-, CelebA-HQ, FFHQ and ImageNet datasets. 
Training can be started by running

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
``` 

EXAMPLE COMMAND:
```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/CUSTOM-ldm-vq-4.yaml -t --gpus 0,1
```

Possible ``<config_spec>`` options also include
- `celebahq-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3)
- `ffhq-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
- `lsun_bedrooms-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
- `lsun_churches-ldm-vq-4`(f=8, KL-reg. autoencoder, spatial size 32x32x4),
- `cin-ldm-vq-8`(f=8, VQ-reg. autoencoder, spatial size 32x32x4).


---
---

# Sampling LDM Model

Once you've downloaded or generated a model checkpoint(s), we also provide a script for sampling from unconditional LDMs. Start it via:

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta> 
```

---
---

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.48550/arxiv.2204.11824,
  doi = {10.48550/ARXIV.2204.11824},
  url = {https://arxiv.org/abs/2204.11824},
  author = {Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and Ommer, Björn},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Retrieval-Augmented Diffusion Models},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}


```


