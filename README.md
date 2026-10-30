# Adaptive Camera Settings

Predicting the camera settings (ISO, shutter speed, aperture) that maximize a downstream classifier's accuracy from a **single auto-exposure image**, so only **two captures** are needed instead of an exhaustive 27-shot sweep.

> Poster at DAGM GCPR 2026, University of Siegen.
> Authors: Kalyani Rameshan Nambiar, Jerin Paul, Sanjay Suresh, Jan Philipp Schneider, Michael Moeller

[Paper (PDF)](GCPR___Predicting_Camera_Parameters-3.pdf) | [Reference paper](2503.02170v1.pdf)

---

## Overview

Auto-exposure is tuned for human-pleasing images, not classifier accuracy. Lens (Baek et al., ICLR 2025) improves accuracy by capturing several images and keeping the one with the highest softmax confidence, but this needs 6 to 27 captures per scene.

We train a lightweight policy network that predicts the optimal setting directly from the auto-exposure (AE) image, giving a two-capture pipeline: one AE capture, then one capture with the predicted settings.

<!-- TODO: add Figure 1 from the poster -->
<!-- ![Overview](output/overview.png) -->

## Methods

| Method | Input | Loss |
|---|---|---|
| **LACE** | ResNet-18 features + 6-d lighting vector | Class-balanced cross-entropy with label smoothing (alpha = 0.05) |
| **LAGE** | ResNet-18 features + 6-d lighting vector | Hinge loss with margin m = 1 |
| **HB** | ResNet-18 features + 74-d patch-wise histogram | Huber loss (regression on setting index) |

All variants use a 2-layer MLP head over N = 27 candidate settings, followed by a frozen classifier.

## Results

ResNet-18 accuracy on ImageNet-ES-Diverse:

| Method | Top-1 (%) | Captures |
|---|---|---|
| Naive Auto-Exposure | 12.17 | 1 |
| LACE (Ours) | 28.39 ± 0.35 | 2 |
| LAGE (Ours) | **29.28 ± 0.25** | 2 |
| HB (Ours) | 24.22 ± 1.65 | 2 |
| Lens CSA1 (Random) | 23.33 ± 1.27 | 6 |
| Lens CSA2 (Grid Random) | 23.47 ± 1.23 | 6 |
| Lens CSA3 (Fast Acquisition) | 21.83 | 18 |
| Lens Full | 30.67 | 27 |
| Theoretical Oracle-S (upper bound) | 47.67 | 27 |

Inference takes about 3.5 ms (LACE, LAGE) and 5.15 ms (HB) on a single NVIDIA T4 GPU.

## Repository Structure

```
.
├── data/                                      # Dataset files and metadata
├── model/                                     # Model definitions / checkpoints
├── notebooks/                                 # Experiment notebooks
├── output/                                    # Results and figures
├── src/                                       # Source code
├── GCPR___Predicting_Camera_Parameters-3.pdf  # Our paper
├── environment.yml                            # environment file
└── 2503.02170v1.pdf                           # Reference paper

```

<!-- TODO: add a one-line description of the key files inside src/ and notebooks/ -->

## Installation

```bash
git clone https://github.com/j3rinpaul/Adaptive_Camera_Settings.git
cd Adaptive_Camera_Settings
conda create --file environment.yml
```

<!-- TODO: add requirements.txt and state the Python / PyTorch versions you used -->

## Dataset

Experiments use **ImageNet-ES-Diverse** (Baek et al., CVPR 2024).

- Train: 4,800 scenes (129,600 images)
- Validation: 600 scenes
- Test: 600 scenes (16,200 images, including lighting variations)

<!-- TODO: add download / preparation instructions and where to place files in data/ -->

## Usage


**All the experiements are done using jupyter notebeooks, so run `Lens_testing_regression.ipynb` from notebooks to verify.**

## Future Work

- Evaluate across all classifiers used in Lens
- Full ablation over architectures, input representations, and losses

## Citation

```bibtex
@inproceedings{nambiar2026exposure,
  title     = {Towards Prediction of Optimal Camera Exposure Parameters for Image Classification},
  author    = {Nambiar, Kalyani Rameshan and Paul, Jerin and Suresh, Sanjay and Schneider, Jan Philipp and Moeller, Michael},
  booktitle = {DAGM German Conference on Pattern Recognition (GCPR)},
  year      = {2026}
}
```

<!-- TODO: update once the official citation is available -->

## References

1. E. Baek et al. *Lens: Hardware-aware adaptive sensor control for robust vision.* ICLR, 2025.
2. K. He et al. *Deep residual learning for image recognition.* CVPR, 2016.
3. K. Crammer and Y. Singer. *On the algorithmic implementation of multiclass kernel-based vector machines.* JMLR, 2001.
4. E. Baek et al. *Unexplored faces of robustness: Covariate shifts in environment and sensor domains.* CVPR, 2024.

## License

<!-- TODO: choose a license, e.g. MIT -->
