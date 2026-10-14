# CucuGAN

Verwandelt Gesichter mit einem selbst implementierten CycleGAN (PyTorch) in Gurkengesichter.
Status: in Arbeit, siehe [PLAN.md](PLAN.md).

## Daten

Die Bilder sind nicht im Repository enthalten. Sie liegen lokal unter `data/raw/`.

### Domäne A: Gesichter
| Datensatz | Quelle | Lizenz | Ordner |
|---|---|---|---|
| FFHQ small (Teilmenge von FFHQ, Karras et al. 2019) | [Kaggle](https://www.kaggle.com/datasets/tommykamaz/faces-dataset-small), Original: [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) | CC BY-NC-SA 4.0 | `data/raw/ffhq/` |

### Domäne B: Gurken
| Datensatz | Autor:innen | DOI | Lizenz | Ordner |
|---|---|---|---|---|
| CUCUMBER (2024) | S. Saha, T. Sarkar | [10.17632/zncrtr2yhn.1](https://doi.org/10.17632/zncrtr2yhn.1) | CC BY 4.0 | `data/raw/cucumber/saha2024/` |
| Good and Bad classification of Cucumber (2025) | S. Sarkar, T. Sarkar | [10.17632/9rs9vmm2bv.1](https://doi.org/10.17632/9rs9vmm2bv.1) | CC BY 4.0 | `data/raw/cucumber/sarkar2025/` |
| Cucumber Dataset (2023) | T. Jin | [10.5281/zenodo.10081197](https://doi.org/10.5281/zenodo.10081197) | CC BY 4.0 | `data/raw/cucumber/jin2023/` |

Verwendet werden nur die gesunden Gurken („Good“/„good“) bzw. bei Jin nur der Ordner „Fruit Dataset“.
