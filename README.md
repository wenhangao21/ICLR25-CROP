# [ICLR 2025] Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators

This repository contains the code implementation for the paper titled "Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators", **W.Gao**, R.Xu, Y.Deng, Y.Liu. The Thirteenth International Conference on Learning Representations (ICLR), 2025

Full paper on [OpenReview](https://openreview.net/forum?id=J9FgrqOOni).

<p align="center">
  <img src="https://wenhangao21.github.io/images/ICLR_DME.png" alt="Figure" width="400"/>
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
This paper studies the discretization-invariance property (i.e., the ability to perform zero-shot super-resolution tasks) of grid-based neural operators, using Fourier Neural Operators (FNO) as an example. Cross-resolution (super-resolution) capability is a desirable feature of neural operators like FNO. However, there is a price to pay. It is a long-standing misconception in the community that grid-based neural operators can perform super-resolution tasks without any degradation in performance. We clarify this misconception by identifying the root cause of such degradation: **Discretization Mismatch Errors (DMEs)**.
- We define discretization mismatch errors (DMEs) in Sec. 4.1 as the difference between the neural operator outputs when taking the same input at different discretizations.
- We provide an upper bound on the DME for grid-based neural operators in Sec. 4.2. 
	- Given a fixed training resolution and a higher testing resolution, we show that the DME increases as the testing resolution increases. 
	- Moreover, this error accumulates through the layers of the neural operator and across autoregressive time steps (if applicable).
- We verify our theoretical conclusions through empirical experiments in Sec. 5.1.1 (see Fig. 3).
- We propose a simple solution in Sec. 4.3 and demonstrate its effectiveness in Sec. 5.1.2. This simple solution restricts the latent feature functions to be in a bandlimited function space. By doing so, there are two main benefits:
	- Since the latent feature functions reside in a bandlimited function space, we can fix the resolution of their discretized forms within the limits tolerated by the bandlimits. As a result, no discretization mismatch errors are introduced by the intermediate layers.
	- There are no aliasing errors (see the Convolutional Neural Operator paper by Bogdan Raonić et al.).
- However, we advocate for further studies in this direction to explore more advanced and principled approaches, potentially informed by known physics, which we discuss in Appendix E.


## Preparing Data

The data used in this paper can be downloaded at *To be added*.

All PDE data are directly adopted from previously established open-source datasets or generated using publicly available scripts. Please see Appendix D of the paper for details.

## How to Run

The commands for training are provided below. Note that there are different choices for the intermediate neural operator. We provide three different intermediate neural operators; however, any architecture can be used. Our findings indicate that the choice of the intermediate operator is crucial.

```python
To_be_added
```

- We also provide the uncleaned original code and pretrained  that can produce the exact same results as in the paper under the folder

## Structure

To be added

