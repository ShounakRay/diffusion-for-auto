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
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm-shounak
```

## Unconditional Models

We also provide a script for sampling from unconditional LDMs (e.g. LSUN, FFHQ, ...). Start it via

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sample_diffusion.py -r models/ldm/<model_spec>/model.ckpt -l <logdir> -n <\#samples> --batch_size <batch_size> -c <\#ddim steps> -e <\#eta> 
```

# Train your own LDMs

## Data preparation

#### Donwload the necessary autoencoder
> From root directory of this repository:
wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
cd models/first_stage_models/vq-f4
unzip -o model.zip

## Model Training

Logs and checkpoints for trained models are saved to `logs/<START_DATE_AND_TIME>_<config_spec>`.

### Training autoencoder models

Configs for training a KL-regularized autoencoder on ImageNet are provided at `configs/autoencoder`.
Training can be started by running
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,    
```
where `config_spec` is one of {`autoencoder_kl_8x8x64`(f=32, d=64), `autoencoder_kl_16x16x16`(f=16, d=16), 
`autoencoder_kl_32x32x4`(f=8, d=4), `autoencoder_kl_64x64x3`(f=4, d=3)}.

For training VQ-regularized models, see the [taming-transformers](https://github.com/CompVis/taming-transformers) 
repository.

### Training LDMs 

In ``configs/latent-diffusion/`` we provide configs for training LDMs on the LSUN-, CelebA-HQ, FFHQ and ImageNet datasets. 
Training can be started by running

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
``` 

where ``<config_spec>`` is one of {`celebahq-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),`ffhq-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
`lsun_bedrooms-ldm-vq-4`(f=4, VQ-reg. autoencoder, spatial size 64x64x3),
`lsun_churches-ldm-vq-4`(f=8, KL-reg. autoencoder, spatial size 32x32x4),`cin-ldm-vq-8`(f=8, VQ-reg. autoencoder, spatial size 32x32x4)}.

# Model Zoo 

## Pretrained Autoencoding Models
All models were trained until convergence (no further substantial improvement in rFID).

| Model                   | rFID vs val | train steps           |PSNR           | PSIM          | Link                                                                                                                                                  | Comments              
|-------------------------|------------|----------------|----------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| f=4, VQ (Z=8192, d=3)   | 0.58       | 533066 | 27.43  +/- 4.26 | 0.53 +/- 0.21 |     https://ommer-lab.com/files/latent-diffusion/vq-f4.zip                   |  |
| f=4, VQ (Z=8192, d=3)   | 1.06       | 658131 | 25.21 +/-  4.17 | 0.72 +/- 0.26 | https://heibox.uni-heidelberg.de/f/9c6681f64bb94338a069/?dl=1  | no attention          |
| f=8, VQ (Z=16384, d=4)  | 1.14       | 971043 | 23.07 +/- 3.99 | 1.17 +/- 0.36 |       https://ommer-lab.com/files/latent-diffusion/vq-f8.zip                     |                       |
| f=8, VQ (Z=256, d=4)    | 1.49       | 1608649 | 22.35 +/- 3.81 | 1.26 +/- 0.37 |   https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip |  
| f=16, VQ (Z=16384, d=8) | 5.15       | 1101166 | 20.83 +/- 3.61 | 1.73 +/- 0.43 |             https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1                        |                       |
|                         |            |  |                |               |                                                                                                                                                    |                       |
| f=4, KL                 | 0.27       | 176991 | 27.53 +/- 4.54 | 0.55 +/- 0.24 |     https://ommer-lab.com/files/latent-diffusion/kl-f4.zip                                   |                       |
| f=8, KL                 | 0.90       | 246803 | 24.19 +/- 4.19 | 1.02 +/- 0.35 |             https://ommer-lab.com/files/latent-diffusion/kl-f8.zip                            |                       |
| f=16, KL     (d=16)     | 0.87       | 442998 | 24.08 +/- 4.22 | 1.07 +/- 0.36 |      https://ommer-lab.com/files/latent-diffusion/kl-f16.zip                                  |                       |
 | f=32, KL     (d=64)     | 2.04       | 406763 | 22.27 +/- 3.93 | 1.41 +/- 0.40 |             https://ommer-lab.com/files/latent-diffusion/kl-f32.zip                            |                       |

### Get the models

Running the following script downloads und extracts all available pretrained autoencoding models.   
```shell script
bash scripts/download_first_stages.sh
```

The first stage models can then be found in `models/first_stage_models/<model_spec>`


## Misc.

Colab notebook https://colab.research.google.com/drive/1xqzUi2iXQXDqXBHQGP9Mqt2YrYW6cx-J?usp=sharing

## Comments 

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


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


