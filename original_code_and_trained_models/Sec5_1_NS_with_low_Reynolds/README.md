This directory contains the code to reproduce the results in Sec. 5.1. The original dataset can trained models can be downloaded [here](https://drive.google.com/drive/folders/1OK3VNzrAKS6vEqwo69UMdOc_DAtnRpMl?usp=sharing)

The data can be generated directly using the scripts under the folder `data_generation`, which is from [1]. You can expect to get extremely similar results even from different runs of generated datasets as the variance for the GRF is relatively small as provided in [1].
**Note:** Reference [1] also provides pre-generated data. However, we found that the data provided by them do not come from the same distribution as in the provided scripts. Testing on the script-generated data results in significantly bad performance.

- Table 2 and Fig. 1 (a)
	-  **Table 2 FNO and Fig. 1 (a):** The folder `FNO` contains the training scripts for FNO, along with the trained models. These scripts are directly adopted from [1] without any modifications to the architecture or training, except that the teacher forcing strategy is applied [2].
	- **Table 2 Alias-free FNO 2×:** The folder `Alias_Free_FNO_2X` contains the training scripts for FNO with 2x up and down samplings in the activation functions, along with the trained models.
	- **Table 2 Alias-free FNO Fixed:** The folder `Alias_Free_FNO_Fixed` contains the training scripts for FNO with fixed resolution samplings in the activation functions, along with the trained models.
	- **Table 2 CNO with interpolation:** The folder `CNO_with_interpolation` contains the training scripts for CNO [3] with bicubic interpolation applied for cross-resolution tasks, along with the trained models.
	- **Table 2 CROP:** The folder `CROP` contains the training scripts for CROP along with the trained models. The P layer combined with the first Fourier layer can be seen as the CROP lifting layer. Since the resolution is fixed to respect the band limits, we apply a residual connection only to the Fourier modes within the band limits. This entire process can be simplified by CROPping out the high-frequency modes at the beginning for computational efficiency. The same applies to the projection layer.

The experiments are repeated 10 times for all FNO-based models; they are repeated 5 times for CNO as CNO is much more computationally expensive and was added during rebuttal. A sample shell script:
 ```
for seeds in 0 1 2 3 4 5 6 7 8 9
do
    python3 fno.py --seed=$seeds
done
``` 

- Fig. 1 (b)
	- The folder `Increase_with_layers` contains the scripts for measuring all the errors. This experiments are repeated 20 times on a much smaller set of training data. The errors are also measured on the training set sa the purpose is only to show the discretization error. The trained models (300 of them) can be downloaded via the link above. 



[1] Fourier Neural Operator for Parametric Partial Differential Equations, Zongyi Li et al., ICLR 2021

[2] Factorized Fourier Neural Operators, Alasdair Tran et al., ICLR 2023

[3] Convolutional Neural Operators for robust and accurate learning of PDEs, Bogdan Raonić, et al., NeurIPS 2023