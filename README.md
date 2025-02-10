# [ICLR 2025] Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators
<div align="center">
[![Paper](https://img.shields.io/badge/PDF-Paper-red.svg)](https://openreview.net/forum?id=J9FgrqOOni)
[![Code](https://img.shields.io/badge/RISE-Code-orange.svg)]([https://github.com/QuJX/RISE](https://github.com/wenhangao21/ICLR25-CROP))
</div>

This repository contains the code implementation for the paper titled "Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators"

**[Wenhan Gao](https://wenhangao21.github.io/)**, [Ruichen Xu](https://calendar.stonybrook.edu/site/iacs/event/iacs-student-seminar-speaker-ruichen-xu-dept-of-applied-mathematics--statistics/), [Yuefan Deng](https://www.stonybrook.edu/commcms/ams/people/_faculty_profiles/deng), [Yi Liu](https://jacoblau0513.github.io/)

The Thirteenth International Conference on Learning Representations (ICLR), 2025

Full paper on [OpenReview](https://openreview.net/forum?id=J9FgrqOOni).

<p align="center">
  <img src="https://wenhangao21.github.io/images/ICLR_DME.png" alt="Figure" width="550"/>
</p>

**Bibtex**:
```bibtex
@inproceedings{gao2025discretizationinvariance,
	title={Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators},
	author={Wenhan Gao and Ruichen Xu and Yuefan Deng and Yi Liu},
	booktitle={The Thirteenth International Conference on Learning Representations},
	year={2025},
	url={https://openreview.net/forum?id=J9FgrqOOni}
}
```

## Introduction 
> This paper studies the discretization-invariance property (i.e., the ability to perform zero-shot super-resolution tasks) of grid-based neural operators, using Fourier Neural Operators (FNO) as an example. **Cross-resolution (super-resolution) capability is a desirable feature of neural operators like FNO. However, there is a price to pay.** It is a long-standing misconception in the community that grid-based neural operators can perform super-resolution tasks without any degradation in performance. We clarify this misconception by identifying the root cause of such degradation: **Discretization Mismatch Errors (DMEs)**.
- We define discretization mismatch errors (DMEs) in Sec. 4.1 as the difference between the neural operator outputs when taking the same input at different discretizations.
- We provide an upper bound on the DME for grid-based neural operators in Sec. 4.2. 
	- Given a fixed training resolution and a higher testing resolution, we show that the DME increases as the testing resolution increases. 
	- Moreover, this error accumulates through the layers of the neural operator and across autoregressive time steps (if applicable).
- We verify our theoretical conclusions through empirical experiments in Sec. 5.1.1 (see Fig. 3).
- We propose a simple solution in Sec. 4.3 and demonstrate its effectiveness in cross-resolutiont tasks in Sec. 5.1.2. This simple solution restricts the latent feature functions to be in a bandlimited function space. By doing so, there are three main benefits:
	- Since the latent feature functions reside in a bandlimited function space, we can fix the resolution of their discretized forms within the limits tolerated by the bandlimits. As a result, no discretization mismatch errors are introduced by the intermediate layers.
	- There are no aliasing errors (see the Convolutional Neural Operator paper by Bogdan Raonić et al.).
	- As the intermediate resolution is fixed, we can use any fixed-size network components, e.g. fixed-size local convolutions. 
- However, we advocate for further studies in this direction to explore more advanced and principled approaches, potentially informed by known physics, which we discuss in Appendix E.


## How to Run

While various intermediate neural operators can be used, we have tested a few examples, including FNO, FNO with local fixed-size convolutions, U-Net (CNO without up and downsampling in activation), and a simple CNN with Fourier layers for global learning. We find that the choice of the intermediate operator is crucial; however, designing optimal architectures is beyond the scope of this work. **Rather than proposing a specific architecture, our goal is to encourage the development of more advanced methods for cross-resolution applications. We advocate for architectures that balance strong performance with physical principles, even if they are not necessarily suited for super-resolution tasks.**

A sample CROP pipeline is provided, where the intermediate neural operator is an U-Net from CNO [1] with the anti-alias activation removed. Use the following command to run the train script (CRNO: Cross-Resolution Neural Operator).

```
python3.12 train.py --which_example ns_high_r --which_model CRNO --seed 42
```

## Original Code and Trained Models

We provide the original, uncleaned (but still readable) code along with some pretrained models. These are available in the `original_code_and_trained_models` folder.

## Preparing Data

The data used in this paper can be downloaded [here](https://drive.google.com/drive/folders/1OK3VNzrAKS6vEqwo69UMdOc_DAtnRpMl?usp=sharing). The descriptions and usage are provided in their respective sections under `original_code_and_trained_models`.

The Darcy flow data (non-linear mapping) can be downloaded [here](https://drive.google.com/file/d/1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf/view?usp=sharing), which is uploaded by the authors of FNO [2].

**All PDE data are adopted from previously established open-source datasets or generated using publicly available scripts. Please see Appendix D of the paper for details.**

[1] Convolutional Neural Operators for robust and accurate learning of PDEs, Bogdan Raonić, et al., NeurIPS 2023

[2] Fourier Neural Operator for Parametric Partial Differential Equations, Zongyi Li et al., ICLR 2021


</div>
<p align="center"> 
  Visitor count<br>
  <img src="https://profile-counter.glitch.me/wenhangao21-ICLR25-CROP/count.svg" />
</p>
