# Results









#### RobotCar Dataset

| Alg                                             | ad   | x    | y    | fd   | v    |
| ----------------------------------------------- | ---- | ---- | ---- | ---- | ---- |
| VTGNet                                          | 0.88 | 0.78 | 0.28 | 1.77 | 0.56 |
| CICT                                            | 0.76 | 0.55 | 0.26 | 1.43 | 0.42 |
| DIM (single model, output 4 points)             | 0.47 | 0.37 | 0.20 | 1.14 | -    |
| DIM (single model, output 16 points)            | 0.92 | 0.61 | 0.46 | 2.46 | -    |
|                                                 |      |      |      |      |      |
| Single-Image-CT (Avg 5 models)                  | 0.80 | 0.61 | 0.35 | 2.13 |      |
| Single-Image-CT (Min uncertainty)               | 0.96 | 0.73 | 0.44 | 2.53 |      |
| Single-Image-CT+discriminator (Single model)    | 0.97 | 0.66 | 0.55 | 2.17 |      |
| Single-Image-CT+discriminator (Avg 5 models)    | 0.80 | 0.61 | 0.33 | 2.18 |      |
| Single-Image-CT+discriminator (Min uncertainty) | 0.85 | 0.62 | 0.40 | 2.25 |      |
|                                                 |      |      |      |      |      |
|                                                 |      |      |      |      |      |
|                                                 |      |      |      |      |      |

### 测试模型泛化能力(训练集到OOD数据) RobotCar2KITTI

| Alg                                               | ad   | x    | y    | fd   | v    |
| ------------------------------------------------- | ---- | ---- | ---- | ---- | ---- |
| DIM (single model, output 4 points)               | 2.45 | 1.74 | 1.23 | 5.85 | -    |
| DIM (single model, output 16 points)              | 2.81 | 1.84 | 1.74 | 6.76 | -    |
|                                                   |      |      |      |      |      |
| Single-Image-CT (Avg 5 models)                    | 2.52 | 1.66 | 1.50 | 6.38 |      |
| Single-Image-CT (Min uncertainty)                 | 3.04 | 1.85 | 1.92 | 7.57 |      |
| Single-Image-CT+discriminator (Single model)      | 2.80 | 1.62 | 1.83 | 5.49 |      |
| Single-Image-CT+discriminator (Avg 5 models)      | 2.05 | 1.29 | 1.27 | 5.46 |      |
| Single-Image-CT+discriminator (Min uncertainty)   | 2.10 | 1.40 | 1.14 | 5.39 |      |
| RIP (5 models, output 4 points, Worst Case Model) | 2.09 | 1.58 | 0.96 | 5.14 |      |
|                                                   |      |      |      |      |      |
| GAN-search-vector64                               | 0.59 | 0.40 | 0.37 | 1.17 |      |
| GAN-search-vector16                               | 0.97 | 0.39 | 0.78 | 2.32 |      |















#  New

* RIP: 跑 RIP/train_dim_multi_model.py 测test_dim_multi_model.py
* SICT-avg/min 跑/测 Ours/train_path-oxford.py
* IL+GAN+disc 跑/测 Oursv2/train_IL_GAN.py
* SICT+disc 跑/测 Ours/train_adversarial-single-model-bin-class.py
* 