### AI-Generated Product Image Detection for E-Commerce Trust & Safety
ListingAuthentic is an end-to-end computer vision pipeline for distinguishing **AI-generated product images** from **authentic product photographs** in online marketplaces such as Etsy.

It combines **ConvNeXt V2**, **EfficientNetV2-S**, ensemble learning, threshold calibration, and test-time augmentation to identify synthetic product imagery — flagging suspicious listings for review while balancing false positives against missed detections.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Key Results](#key-results)
- [Why an Ensemble?](#why-an-ensemble)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Example Decision Flow](#example-decision-flow)
- [Error Analysis](#error-analysis)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Technology Stack](#technology-stack)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team Contributions](#team-contributions)
- [Citation](#citation)
- [Disclaimer](#disclaimer)

---

## Problem Statement

Product images strongly influence buyer decisions on online marketplaces. However, generative AI systems can produce highly realistic product mockups that do not accurately represent the physical item a buyer will receive.

This can result in:

- Misleading product listings
- Buyers receiving prints, low-quality goods, or mass-produced items
- Reduced confidence in handmade marketplaces
- Increased complaints, refunds, and moderation workload
- Genuine sellers being incorrectly flagged by overly aggressive detectors

ListingAuthentic treats detection as a binary image-classification problem:

```text
Input image → AI-generated or authentic
```

The system focuses on realistic e-commerce imagery, where synthetic artifacts can be subtle and traditional visual inspection may be unreliable.

---

## Key Results

| Model configuration | F1 score | Precision | Recall |
|---|---:|---:|---:|
| Etsy baseline | 0.6359 | — | — |
| ConvNeXt V2 | 0.8778 | 0.8584 | 0.8981 |
| EfficientNetV2-S | 0.8794 | 0.8895 | 0.8695 |
| Basic ensemble | 0.8924 | 0.8907 | 0.8942 |
| **Final ensemble (TTA + threshold tuning)** | **0.8950** | 0.8848 | **0.9054** |

The final model improved the F1 score from a baseline of 0.6359 to **0.8950** — an absolute improvement of **0.2591** — demonstrating the value of combining models with different error patterns.

On the evaluation set, the final ensemble correctly classified **430 authentic images** and **419 AI-generated images**. Remaining errors: 67 authentic images misclassified as AI-generated, and 44 AI-generated images misclassified as authentic.

---

## Why an Ensemble?

The two models learned complementary visual representations:

| Model | Strength |
|---|---|
| **ConvNeXt V2** | Higher recall — catches more AI-generated images |
| **EfficientNetV2-S** | Higher precision — fewer false-positive AI predictions |
| **Ensemble** | Combines both for a more balanced decision boundary |

This mattered because a marketplace moderation system must control both error types:

- **False positives** can unfairly affect genuine sellers
- **False negatives** allow misleading synthetic listings to remain online

Model diversity contributed more to final performance than optimizing a single architecture in isolation.

---

## System Architecture

```text
                    ┌───────────────────────┐
                    │  Product listing image │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Image preprocessing  │
                    │   Resize + normalize   │
                    └───────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       ┌──────────────────┐            ┌───────────────────┐
       │   ConvNeXt V2     │            │ EfficientNetV2-S  │
       │  Recall-focused   │            │ Precision-focused │
       └─────────┬─────────┘            └─────────┬─────────┘
                 │                                │
                 └────────────────┬───────────────┘
                                  ▼
                     ┌───────────────────────┐
                     │  Prediction ensemble   │
                     │  Probability fusion    │
                     └───────────┬───────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Threshold calibration  │
                     └───────────┬───────────┘
                                 ▼
                ┌────────────────┴────────────────┐
                ▼                                  ▼
        Authentic product                  AI-generated image
        Accept or monitor                  Flag for review
```

---

## Methodology

### Data preparation

The dataset contains labelled training images and unlabelled test images. Image identifiers were matched with file paths, while class labels were loaded from a CSV file. The project used **5-fold cross-validation** to improve reliability and reduce dependence on a single train-validation split.

Preprocessing included:

- Image resizing for each model architecture
- ImageNet mean and standard-deviation normalization
- Training and validation separation
- Label verification
- Batch-based loading through PyTorch
- Unchanged validation images for fair evaluation

### Data augmentation

Training augmentation included random horizontal flipping, random cropping, resizing, and model-specific normalization — encouraging the models to learn product-image characteristics that remain stable under small changes in viewpoint, framing, and composition.

### ConvNeXt V2

Used as a strong visual feature extractor; its prediction behavior favored recall.

```text
F1 score: 0.8778   Precision: 0.8584   Recall: 0.8981
```

### EfficientNetV2-S

Selected for its balance between model capacity and computational efficiency; produced more conservative predictions and achieved higher precision.

```text
F1 score: 0.8794   Precision: 0.8895   Recall: 0.8695
```

### Training strategy

Both networks were fine-tuned from pretrained weights using:

- PyTorch
- AdamW optimization
- Binary cross-entropy loss with label smoothing
- Experimentally tuned batch sizes
- Validation-based model selection
- 5-fold cross-validation

Training from scratch was not selected because the available dataset size was better suited to transfer learning.

### Ensemble strategy

The ensemble combines prediction probabilities from ConvNeXt V2 and EfficientNetV2-S, benefiting from both recall-oriented and precision-oriented predictions.

```text
F1 score: 0.8924   Precision: 0.8907   Recall: 0.8942
```

### Threshold tuning

A default decision threshold does not necessarily provide the best precision/recall balance. The classification threshold was adjusted using validation predictions to control the trade-off between missing AI-generated images and incorrectly flagging authentic photographs.

### Test-time augmentation

Test-time augmentation (TTA) applies multiple lightweight transformations to an evaluation image and combines the resulting predictions, providing an additional improvement after the ensemble stage:

```text
F1 score: 0.8950   Precision: 0.8848   Recall: 0.9054
```

The larger performance gain came from the ensemble itself, with TTA and threshold tuning providing additional refinement.

---

## Example Decision Flow

```python
convnext_probability = convnext_model(image)
efficientnet_probability = efficientnet_model(image)

ensemble_probability = (
    convnext_probability + efficientnet_probability
) / 2

if ensemble_probability >= decision_threshold:
    decision = "flag_for_review"
else:
    decision = "authentic_or_monitor"
```

The exact fusion method and threshold should be selected using validation data and the marketplace's operational priorities.

---

## Error Analysis

Main error cases:

- Highly realistic AI-generated images that closely resemble photographs
- Authentic images with poor lighting, compression, blur, or visual noise
- Product photographs with unusual backgrounds or heavy editing
- Images whose product category is underrepresented in the training data

These cases are why the model should be used as a **moderation-support system**, not the sole authority for seller enforcement.

---

## Project Structure

```text
ListingAuthentic/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── ensemble_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── convnextv2.py
│   │   ├── efficientnetv2.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── train_convnext.py
│   │   ├── train_efficientnet.py
│   │   └── cross_validation.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── threshold_tuning.py
│   │   └── confusion_matrix.py
│   └── inference/
│       ├── predict.py
│       └── test_time_augmentation.py
├── visualizations/
│   ├── confusion_matrix.png
│   ├── f1_comparison.png
│   └── precision_recall_comparison.png
├── weights/
│   ├── convnextv2_fold_*.pth
│   ├── efficientnetv2_fold_*.pth
│   └── ensemble_config.json
├── requirements.txt
├── train.py
├── predict.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/listingauthentic.git
cd listingauthentic

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

**1. Prepare the dataset**

```bash
python -m src.data.preprocessing \
  --input-dir data/raw \
  --output-dir data/processed
```

**2. Train ConvNeXt V2**

```bash
python -m src.training.train_convnext \
  --data-dir data/processed \
  --folds 5
```

**3. Train EfficientNetV2-S**

```bash
python -m src.training.train_efficientnet \
  --data-dir data/processed \
  --folds 5
```

**4. Evaluate individual models**

```bash
python -m src.evaluation.metrics \
  --model convnextv2 \
  --data-dir data/processed/validation
```

**5. Tune the ensemble threshold**

```bash
python -m src.evaluation.threshold_tuning \
  --predictions outputs/ensemble_predictions.csv
```

**6. Evaluate the final ensemble**

```bash
python -m src.evaluation.ensemble_evaluation \
  --predictions outputs/ensemble_predictions.csv \
  --threshold outputs/best_threshold.json
```

---

## Inference

Run prediction on a single image:

```bash
python predict.py \
  --image examples/product_image.jpg \
  --model-dir weights/ \
  --use-tta
```

Example output:

```json
{
  "image": "examples/product_image.jpg",
  "prediction": "AI-generated",
  "synthetic_probability": 0.9054,
  "decision_threshold": 0.5,
  "review_required": true
}
```

For marketplace deployment, the output should be interpreted as a risk signal:

| Probability | Action |
|---|---|
| Low | Accept automatically or continue monitoring |
| Intermediate | Send to secondary checks |
| High | Flag for trust-and-safety review |

---

## Technology Stack

Python · PyTorch · Torchvision · ConvNeXt V2 · EfficientNetV2-S · Scikit-learn · NumPy · Pandas · OpenCV · Matplotlib · Seaborn · CUDA-compatible GPU support

---

## Limitations

- The model uses image information only — listing metadata, descriptions, seller history, and buyer reports are not included
- Dataset size may limit generalization to new product categories
- New image-generation models may produce artifacts not represented in training
- Image compression, screenshots, and post-processing can affect predictions
- Ensemble inference requires more computational resources than a single model
- Results may vary across image domains, cameras, lighting conditions, and marketplace categories
- A high synthetic-image probability is **not proof** that a seller acted dishonestly

In production, automated predictions should be combined with human review, seller appeals, and additional signals before serious enforcement action is taken.

---

## Future Work

- Larger and more diverse e-commerce datasets
- Multimodal learning using image, title, description, and metadata
- Marketplace-specific calibration by product category
- Model monitoring for distribution shift
- Continuous retraining using reviewer feedback
- Image provenance and metadata analysis
- Duplicate and near-duplicate listing detection
- Detection of AI-generated product descriptions
- Faster model distillation for real-time inference
- Human-in-the-loop moderation dashboards
- Robustness evaluation against image compression and adversarial editing
- Support for emerging image-generation models

---

## Team Contributions

**Divyansh Doshi**
- Implemented and optimized the EfficientNetV2-S model
- Conducted threshold tuning and performance evaluation
- Led the ensemble strategy for combining model predictions
- Contributed to the transfer-learning and evaluation pipeline

**Amisha Sanjay Kadukar**
- Implemented and fine-tuned the ConvNeXt V2 model
- Designed and evaluated test-time augmentation experiments
- Contributed to data preprocessing and augmentation
- Developed visualizations for model performance

**Joint contributions**
- Designed the overall transfer-learning pipeline
- Developed the evaluation framework
- Analysed the ensemble results
- Prepared the final report and project documentation

---

## Citation

If you use this project or build upon the methodology, please cite:

```bibtex
@article{kadukar_doshi_listingauthentic,
  title       = {Ensemble Learning for Detection of AI-Generated Product Images Using ConvNeXt V2 and EfficientNetV2},
  author      = {Kadukar, Amisha Sanjay and Doshi, Divyansh},
  institution = {Dublin City University},
  year        = {2026}
}
```

---

## Disclaimer

ListingAuthentic is intended for research, educational, and trust-and-safety experimentation. It should not be used as the sole basis for permanently suspending sellers, rejecting listings, or making other high-impact decisions without human review, evidence verification, and a fair appeals process.
