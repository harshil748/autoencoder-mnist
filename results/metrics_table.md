# Metrics Table

| Model                  | Latent_Dim | Regularization                           |   Test_MSE |
| :--------------------- | ---------: | :--------------------------------------- | ---------: |
| BasicAutoencoder       |         16 | None                                     | 0.00796162 |
| BasicAutoencoder       |         16 | L2 weight_decay=1e-4                     |  0.0188657 |
| RegularizedAutoencoder |         16 | Dropout p=0.2                            | 0.00891642 |
| DenoisingAutoencoder   |         16 | Gaussian noise                           | 0.00991748 |
| DenoisingAutoencoder   |         16 | Gaussian noise + L2 weight_decay=1e-4    |  0.0208239 |
| DenoisingAutoencoder   |         16 | Salt-pepper noise                        |  0.0134656 |
| DenoisingAutoencoder   |         16 | Salt-pepper noise + L2 weight_decay=1e-4 |  0.0256998 |
| BasicAutoencoder       |         32 | None                                     | 0.00504881 |
| BasicAutoencoder       |         32 | L2 weight_decay=1e-4                     |  0.0197563 |
| RegularizedAutoencoder |         32 | Dropout p=0.2                            | 0.00610709 |
| DenoisingAutoencoder   |         32 | Gaussian noise                           | 0.00759948 |
| DenoisingAutoencoder   |         32 | Gaussian noise + L2 weight_decay=1e-4    |  0.0214026 |
| DenoisingAutoencoder   |         32 | Salt-pepper noise                        |  0.0116596 |
| DenoisingAutoencoder   |         32 | Salt-pepper noise + L2 weight_decay=1e-4 |  0.0259606 |
| BasicAutoencoder       |         64 | None                                     | 0.00427276 |
| BasicAutoencoder       |         64 | L2 weight_decay=1e-4                     |  0.0187604 |
| RegularizedAutoencoder |         64 | Dropout p=0.2                            | 0.00575142 |
| DenoisingAutoencoder   |         64 | Gaussian noise                           | 0.00695945 |
| DenoisingAutoencoder   |         64 | Gaussian noise + L2 weight_decay=1e-4    |  0.0211654 |
| DenoisingAutoencoder   |         64 | Salt-pepper noise                        |  0.0112652 |
| DenoisingAutoencoder   |         64 | Salt-pepper noise + L2 weight_decay=1e-4 |  0.0262311 |
| PCA                    |         32 | Linear PCA baseline                      |  0.0168326 |
