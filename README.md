# unet_SegCLR
The code for the paper Unsupervised Domain Adaptation with Contrastive Learning for OCT Segmentation https://arxiv.org/abs/2203.03664.
It is applied on the BraTS 21 Dataset and the Retinal Datasets (refuge, idrid, rimone).
| D0->D1 CL          |                         | Dt    | Ds    |       |       | D0  | refuge  |
|--------------------|-------------------------|-------|-------|-------|-------|-----|---------|
| supervised         | UpperBound[Dt] (idrid)  | 90.09 | 30.30 |       |       | D1  | idrid   |
|                    | Baseline[Ds] (refuge)   | 73.97 | 95.45 |       |       | D2  | rimone  |
| seed = 1, All      | lambda                  | 10000 |       | 1000  |       | 100 |         |
| intra-domain       | [Gomariz]  8Ds+8Dt      | 82.56 | 93.72 | 78.83 | 95.74 |     |         |
| only-source-domain | Proposed                | 78.93 | 95.40 | 80.06 | 95.50 |     |         |
|                    |                         |       |       |       |       |     |         |
| D0->D2 CL          |                         | Dt    | Ds    |       |       |     |         |
| supervised         | UpperBound[Dt] (rimone) | 92.62 | 48.91 |       |       |     |         |
|                    | Baseline[Ds] (refuge)   | 47.52 | 95.45 |       |       |     |         |
| seed = 1, All      | lambda                  | 10000 |       | 1000  |       | 100 |         |
| intra-domain       | [Gomariz]  8Ds+8Dt      | 61.66 | 94.31 | 64.42 | 94.84 |     |         |
| only source domain |                         | 45.13 | 95.40 | 39.33 | 95.50 |     |         |
