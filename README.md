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
| Cucumber Dataset (2023) | T. Jin | [10.5281/zenodo.10081197](https://doi.org/10.5281/zenodo.10081197) | CC BY 4.0 | `data/raw/cucumber/jin2023/` |

Verwendet wird nur der Ordner „Fruit Dataset“ (ganze Gurken mit Schale), nicht „Flesh DataSet“ (Schnittflächen).
