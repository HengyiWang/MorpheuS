# MorpheuS: Neural Dynamic 360° Surface Reconstruction from Monocular RGB-D Video
### [Paper](https://arxiv.org/abs/2312.00778) | [Project Page](https://hengyiwang.github.io/projects/morpheus) | [Video](https://hengyiwang.github.io/projects/morpheus/videos/morpheus_video_1080p.mp4)

> MorpheuS: Neural Dynamic 360° Surface Reconstruction from Monocular RGB-D Video <br />
> [Hengyi Wang](https://hengyiwang.github.io/), [Jingwen Wang](https://jingwenwang95.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<br />
> CVPR 2024

<p align="center">
  <a href="">
    <img src="./media/morpheus_teaser.gif" alt="Logo" width="80%">
  </a>
</p>



This repository contains the code for the paper MorpheuS: Neural Dynamic 360° Surface Reconstruction from Monocular RGB-D Video, a dynamic surface reconstruction method that leverages a diffusion prior to achieve 360° surface reconstruction.



## Update

- [x] Code for visualization of the results [2024-3-25]
- [x] Code for data pre-processing [2024-3-21]
- [x] Code for MorpheuS [2024-3-18]

## Installation

For detailed instructions, please refer to [INSTALL.md](./docs/INSTALL.md).

## Dataset

Please download here: [Google Drive](https://drive.google.com/drive/folders/1mDuIauf-jxVvvAuefWTt8zK2mUKp_UFr?usp=sharing). Alternatively, use our download script:

```sh
bash scripts/download_data.sh
```

To create your own dataset, please refer to [preprocess](./preprocess/).

## Run

You can run MorpheuS using the code below:

```sh
python morpheus.py --config './configs/snoopy.yaml'
```

For visualizing the results, you can use:

```sh
python visualizer.py --config './configs/snoopy.yaml'
```

## Advanced tips

I have included my tips in the comments prefixed with "NOTE". Given the extensive complexity of this project and the numerous experiments conducted, reading these tips is strongly encouraged to gain a deeper understanding.

Please be aware that this project has undergone significant refactoring. As a result, certain sections may differ from the original codebase. However, these modifications are intended to enhance overall performance and results. If anything is broken, don't hesitate to open an issue:)



## Acknowledgement

We have borrowed codes from following awesome repositories, many thanks to authors for sharing their code:

- [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
- [NDR](https://github.com/USTC3DV/NDR-code)
- [VolSDF](https://github.com/lioryariv/volsdf)

The research presented here has been supported by a sponsored research award from Cisco Research and the UCL Centre for Doctoral Training in Foundational AI under UKRI grant number EP/S021566/1. This project made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1).



## Citation

If you find our code or paper useful for your research, please consider citing:

```
@article{wang2023morpheus,
        title={MorpheuS: Neural Dynamic 360 $\{$$\backslash$deg$\}$ Surface Reconstruction from Monocular RGB-D Video},
        author={Wang, Hengyi and Wang, Jingwen and Agapito, Lourdes},
        journal={arXiv preprint arXiv:2312.00778},
        year={2023}
}
```

