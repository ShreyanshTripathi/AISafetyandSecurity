# TML25_A4_03

# Model Explainability Comparison: LIME vs CAM Methods

This repository contains a comprehensive comparison of different model explainability techniques for image classification, specifically comparing LIME (Local Interpretable Model-agnostic Explanations) with various Class Activation Mapping (CAM) methods.

## Project Structure

```
├── lime.py                 # LIME implementation for image explanations
├── cam.py                  # CAM methods implementation (Grad-CAM, AblationCAM, ScoreCAM)
├── comparison.py           # Comparison analysis between LIME and CAM methods
└── results/                # Output folder containing generated explanations
    ├── overlays/           # Overlaid explanation visualizations
    └── masks/              # Binary/grayscale explanation masks
```


## Features

### LIME Implementation (`lime.py`)

- Batch processing of multiple images
- Organized output structure with separate folders for overlays and masks
- Hyperparameter tuning capabilities for the `explain_instance` function
- Integration with Grad-CAM masks for improved explanations
- Support for reproducible results with proper random seed management


### CAM Methods (`cam.py`)

- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **AblationCAM**: Systematic ablation-based explanations
- **ScoreCAM**: Score-based activation mapping without gradients
- Automatic saving of both overlaid visualizations and raw masks
- Support for ResNet-50 and other (`comparison.py`)
- Side-by-side comparison of LIME and CAM explanations
- Quantitative metrics (IoU, pixel-wise overlap) between different methods
- Hyperparameter optimization using CAM masks as reference
- Statistical analysis of explanation consistency


## Installation

```bash
pip install torch torchvision matplotlib
pip install lime pytorch-grad-cam
pip install scikit-learn scikit-image
pip install pillow numpy pandas
```


## Output Structure

The `results/` folder is organized as follows:

```
results/
├── lime_results/
│   ├── overlays/           # LIME overlaid explanations
│   └── masks/              # LIME binary masks
├── results_gradcam/        # Grad-CAM overlays
├── results_ablationcam/    # AblationCAM overlays  
├── results_scorecam/       # ScoreCAM overlays
├── masks_gradcam/          # Grad-CAM raw masks
├── masks_ablationcam/      # AblationCAM raw masks
└── masks_scorecam/         # ScoreCAM raw masks
```


## Hyperparameter Tuning

The repository includes functionality to tune LIME parameters using Grad-CAM masks as reference:

```python
# Tune LIME parameters for optimal agreement with Grad-CAM
best_params = tune_lime_parameters(
    image_paths=image_paths,
    gradcam_masks_dir='results/masks_gradcam',
    param_grid={
        'num_samples': [500, 1000, 1500],
        'num_features': [5, 10, 15]
    }
)
```


## Technical Details

- **Model**: Pre-trained ResNet-50 on ImageNet
- **Target Layer**: `model.layer4[-1]` for all CAM methods
- **Image Preprocessing**: Standard ImageNet normalization (224×224 resolution)
- **Evaluation Metrics**: IoU (Jaccard Index), pixel-wise overlap, explanation stability


