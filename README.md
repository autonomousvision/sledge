
<p align="center">
    <img alt="sledge" src="assets/sledge_logo_transparent.png" width="500">
    <h1 align="center">Generative Simulation for Vehicle Motion Planning </h1>
    <h3 align="center"><a href="https://arxiv.org/abs/2403.17933">Paper</a> | <a href="https://danieldauner.github.io/assets/pdf/Chitta2024ARXIV_supplementary.pdf">Supplementary</a> | <a href="https://youtu.be/YGxcg2WmkWo?si=-r_6yObHs7H9xJEn&t=28483">Talk</a> </h3>
</p>

<br/>

> [**SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic**](https://arxiv.org/abs/2403.17933) <br>
> [Kashyap Chitta](https://kashyap7x.github.io/), [Daniel Dauner](https://danieldauner.github.io/), and [Andreas Geiger](https://www.cvlibs.net/) <br>
> University of Tübingen, Tübingen AI Center
>
> European Conference on Computer Vision (ECCV), 2024
> <br>
>

This repo contains SLEDGE, the first generative simulator for vehicle motion planning trained on real-world driving logs. We will be publicly releasing our code for simulation, evaluation, and training (including pre-trained checkpoints). 

<br/>

https://github.com/autonomousvision/sledge/assets/50077664/1c653fda-6e44-4018-ae98-2ab3d0439cad

## News &#128240; 
* **`18 Aug, 2024`:** We released v0.1 of the SLEDGE code!
* **`01 Jul, 2024`:** Our paper was accepted at [ECCV 2024](https://eccv.ecva.net/) &#127470;&#127481;
* **`27 Mar, 2024`:** We released our paper on [arXiv](https://arxiv.org/abs/2403.17933)!


## Table of Contents &#128220;
1. [Getting started](#gettingstarted)
2. [Changelog](#changelog)
3. [Contact](#contact)
4. [Citation](#citation)
5. [Other resources](#otherresources)


## Getting started &#128640; <a name="gettingstarted"></a>

- [Installation and download](docs/installation.md)
- [Running the autoencoder](docs/autoencoder.md) 
- [Running the diffusion model](docs/diffusion.md)
- [Simulation and visualization](docs/simulation.md)


## Changelog &#128759; <a name="changelog"></a>
- **`[2024/08/18]`** SLEDGE v0.1 release
  - Scripts for pre-processing and downloads
  - Raster-vector autoencoder (training & latent caching)
  - Latent diffusion models (training & scenario generation)
  - Simple simulations
  - SledgeBoard


## TODO &#128203;<a name="todo"></a>
- [ ] Add videos and talks
- [ ] Release checkpoints
- [ ] Metrics & complete simulation code
- [X] SLEDGE v0.1 & camera ready release
- [x] Initial repository & preprint release


## Contact &#9993; <a name="contact"></a>
If you have any questions or suggestions, please feel free to open an issue or contact us (daniel.dauner@uni-tuebingen.de).

## Citation &#128206; <a name="citation"></a>
If you find SLEDGE useful, please consider giving us a star &#127775; and citing our paper with the following BibTeX entry.

```BibTeX
@InProceedings{Chitta2024ECCV, 
	title = {SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic}, 
	author = {Kashyap Chitta and Daniel Dauner and Andreas Geiger}, 
	booktitle = {European Conference on Computer Vision (ECCV)}, 
	year = {2024}, 
}
```

## Other resources &#128161; <a name="otherresources"></a>

<a href="https://twitter.com/AutoVisionGroup" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Awesome Vision Group?style=social&color=brightgreen&logo=twitter" />
  </a>
<a href="https://twitter.com/kashyap7x" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Kashyap Chitta?style=social&color=brightgreen&logo=twitter" />
  </a>
<a href="https://twitter.com/DanielDauner" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Daniel Dauner?style=social&color=brightgreen&logo=twitter" />
  </a>

- Special thanks to [Agniv Sharma](https://github.com/agshar96) for his reimplementation of [HDMapGen](https://github.com/agshar96/HDMapGen) which we used as a baseline!
- [NAVSIM](https://github.com/autonomousvision/navsim) | [tuPlan garage](https://github.com/autonomousvision/sledge) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)

<p align="right">(<a href="#top">back to top</a>)</p>
