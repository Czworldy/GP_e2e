# Results









#### RobotCar Dataset

| Alg                                               | ad   | x    | y    | fd   | v    |
| ------------------------------------------------- | ---- | ---- | ---- | ---- | ---- |
| VTGNet                                            | 0.88 | 0.78 | 0.28 | 1.77 | 0.56 |
| CICT                                              | 0.76 | 0.55 | 0.26 | 1.43 | 0.42 |
| Single-Image-CT (Min over 5 models)               | 0.87 | 0.62 | 0.45 | 2.29 | -    |
| Single-Image-CT (Min over 5 models)+discriminator | 0.76 | 0.56 | 0.34 | 2.02 | -    |
| DIM (single model, output 4 points)               | 0.47 | 0.37 | 0.20 | 1.14 | -    |
| DIM (single model, output 16 points)              | 0.92 | 0.61 | 0.46 | 2.46 | -    |
|                                                   |      |      |      |      |      |

### 测试模型泛化能力(训练集到OOD数据) RobotCar2KITTI

| Alg                                               | ad   | x    | y    | fd   | v    |
| ------------------------------------------------- | ---- | ---- | ---- | ---- | ---- |
| VTGNet                                            |      |      |      |      |      |
| CICT                                              |      |      |      |      |      |
| Single-Image-CT (Min over 5 models)               | 4.47 | 3.81 | 1.24 | 9.66 | -    |
| Single-Image-CT (Min over 5 models)+discriminator | 1.91 | 1.17 | 1.05 | 4.70 |      |
| DIM (single model, output 4 points)               | 2.45 | 1.74 | 1.23 | 5.85 | -    |
| DIM (single model, output 16 points)              | 2.81 | 1.84 | 1.74 | 6.76 | -    |
|                                                   |      |      |      |      |      |
