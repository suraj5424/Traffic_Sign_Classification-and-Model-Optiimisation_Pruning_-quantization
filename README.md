# Efficient Traffic Sign Classification Using Pruning and Quantization for Embedded Systems

## 🚦 Project Overview

This repository investigates the classification of **German Traffic Signs** using a custom Convolutional Neural Network (CNN), with a particular emphasis on deploying efficient models for **embedded systems** and **edge devices**. The project centers on **model optimization** techniques such as **L1 unstructured pruning** and **Quantization Aware Training (QAT)** to substantially reduce model size and computational demands. The goal is to maintain high classification accuracy while enhancing inference speed and resource efficiency, making the model well-suited for real-time traffic sign recognition on resource-constrained platforms like embedded systems and edge devices.

## ✨ Features

  * **Automated Data Handling:** Downloads and extracts the GTSRB dataset from Kaggle.
  * **Data Augmentation:** Custom pipeline to balance the imbalanced dataset and enhance model robustness via color variations and geometric transformations.
  * **Custom CNN:** A compact and efficient `TrafficSignCNN` architecture for robust traffic sign recognition.
  * **Advanced Model Optimization:**
      * **Pruning:** L1 unstructured pruning implemented via analysis-driven and fixed-percentage strategies.
      * **Quantization Aware Training (QAT):** Integrates quantization into training for improved accuracy retention at lower precision (INT8).
  * **Detailed Model Analysis:** In-depth diagnostics (BN scaling, filter redundancy, activation utility) to guide pruning.
  * **Rigorous Evaluation:** Comprehensive performance assessment using accuracy, precision, recall, F1-score, inference time, GMACs, and model size.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.x
  * Jupyter Notebook or Google Colab (recommended).
  * Kaggle API token (setup instructions provided in the notebook/README).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```
2.  **Open in Google Colab (Recommended):** Upload `[YOUR_NOTEBOOK_NAME].ipynb`.
3.  **Install dependencies:** (Handled by `!pip install` in the notebook, or `pip install torch torchvision pandas matplotlib scikit-learn seaborn ptflops opencv-python Pillow tqdm` locally).

### Usage

1.  **Execute the Jupyter Notebook cells sequentially.**
2.  **Observe Outputs** for data balancing, model analysis, and performance metrics.

-----

## 📂 Project Structure

```
.
├── [Traffic_Sign_Classification_CNN_Pruning_Quantization].ipynb  # Main Jupyter Notebook with all code
├── /content/                   # Working directory for data
│   └── traffic_signal_images/
│       └── Traffic/
│           └── Data/
│               ├── Train/      # Augmented training images (~96,750 images)
│               ├── Test/       # Original external test dataset (12,630 images)
│               ├── Train.csv   # Training metadata (updated after augmentation)
│               └── Test.csv    # Test metadata
└── /content/drive/MyDrive/     # Mounted Google Drive (for Kaggle API key and model saving)
    └── kaggle.json             # Your Kaggle API token
└── /content/models/            # Directory for saved models
    ├── best_baseline.pth       # Saved baseline model checkpoint
    ├── pruned_model_XX.pth     # Saved pruned models (e.g., 10%, 30%, 50%, 90%)
    └── quantized_model.pth     # Saved QAT-trained quantized model
```

-----

## 🔬 Methodology

### Dataset Details

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset:

  * **43 distinct traffic sign classes.**
  * CSV metadata includes `ClassId` and **Region of Interest (ROI) coordinates**.
  * **Highly imbalanced:** Largest class has \~2,250 images; smallest has \~69.
  * **Over 39,000 original training images** (varying sizes from 23x23 to 150x150 pixels, resized to 64x64).
  * **External test dataset of 12,630 images** also with variable dimensions and labels.

### Data Preparation and Augmentation

To address imbalance and enhance generalization, a targeted augmentation strategy was applied:

1.  **Target Balancing:** Each of the 43 classes in the training set was balanced to **2,250 images**.
2.  **Augmentation Strategy:** New images were generated for underrepresented classes by transforming existing ROIs.
3.  **Techniques:**
      * **Color Jitter:** Adjusts brightness, contrast, saturation, and hue.
      * **Affine Transformations:** Randomly shifts images.
      * **Gaussian Blur:** Simulates atmospheric conditions.
4.  **Final Dataset Size:** After augmentation, the training dataset totaled approximately **96,750 images**.

**Result of Data Balancing:** (List of 43 classes each with 2250 images, identical to previous version.)

### Model Architecture: TrafficSignCNN

Our custom `TrafficSignCNN` is designed for efficiency:

  * **Four `conv_block` units:** Each contains `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d` (channels: 32, 64, 128, 256).
  * **Global Average Pooling:** Reduces feature maps for robustness.
  * **Dropout:** Applied before the final linear classifier to prevent overfitting.
  * **QAT Ready:** Includes `QuantStub`, `DeQuantStub`, and `fuse_model` for seamless quantization.

### Model Optimization Pipeline

The optimization pipeline proceeds as: **Baseline Model Training -\> Model Pruning -\> Quantization Aware Training (QAT)**. Two distinct pruning experiments were conducted.

#### 1\. Baseline Model Training

The `TrafficSignCNN` was initially trained (FP32) on the augmented dataset (70% train, 15% validation, 15% test) for 15 epochs using Adam optimizer and CrossEntropyLoss.

**Baseline Model Performance:**

```
Baseline Model Report
Model      | Params    | Test Acc | Size (MB)
-----------|-----------|----------|----------
Baseline   | 399,947   | 0.9974   | 1.61
```

#### 2\. Model Pruning Experiments

Before pruning, a detailed analysis identified suitable layers and parameters for reduction.

**Analysis for Optimization Opportunities (Summary):**
Analysis of BN scaling, filter redundancy, activation utility, and parameter distribution consistently showed significant channel pruning potential (approx. **34-36% of channels** across convolutional layers were low-utility candidates).

**Experiment 1: Analysis-Driven Pruning (Main Pipeline)**
L1 unstructured pruning was applied based on the per-layer percentages identified by the analysis (e.g., 34.4%, 35.9%, 35.9%, 35.9%). The pruned model was then fine-tuned for one epoch.

```
Creating pruned model with channel dimensions:
Original channels: [32, 64, 128, 256]
Pruned channels:   [21, 41, 82, 164]

Pruned Model Report (Analysis-Driven)
Model  | Params  | Test Acc | Size (MB)
-------|---------|----------|----------
Pruned | 167,317 | 0.9974   | 0.68
```

**Experiment 2: Fixed-Percentage Pruning for Performance Observation**
For comparative analysis, pruning was uniformly applied across all layers at fixed percentages: 10%, 30%, 50%, and 90%.

```
================================================================================
Summary of Pruned Models (Fixed-Percentage Experiment)
================================================================================
Prune % | Params  | Test Accuracy | Size (MB) | GMACs (Giga MACs)
--------|---------|---------------|-----------|------------------
10%     | 324,798 | 0.9966        | 1.31      | 0.013          
30%     | 199,356 | 0.9974        | 0.81      | 0.008          
50%     | 103,227 | 0.9908        | 0.42      | 0.004          
90%     | 5,244   | 0.2207        | 0.03      | 0.000          
```

The 30% pruning level offered the best trade-off, maintaining baseline accuracy while significantly reducing parameters.

#### 3\. Quantization Aware Training (QAT)

Instead of post-training quantization, **Quantization Aware Training (QAT)** was used. QAT fine-tunes the model while simulating quantization effects (INT8), allowing it to adapt to lower precision and retain accuracy.

**Methodology for QAT:**

1.  **Module Preparation:** The pruned FP32 model is prepared by fusing `Conv-BatchNorm-ReLU` operations and inserting `QuantStub`/`DeQuantStub` layers and quantization observers.
2.  **Fine-tuning:** The model is fine-tuned with these observers active, allowing it to learn robustness to quantization noise.
3.  **Final Conversion:** After fine-tuning, weights and activations are converted to INT8.

-----

## 📊 Results and Performance

The optimization pipeline successfully reduced model size and computational complexity with minimal accuracy degradation. **All inference prediction results below were obtained on the 12,630 external test images.**

### Overall Model Performance Metrics (Accuracy, Precision, Recall, F1 Score)

| Model     | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :-------- | :----------- | :------------ | :--------- | :----------- |
| Baseline  | 97.65        | 97.71         | 97.65      | 97.59        |
| Pruned(Analysis based pruned)    | 97.18        | 97.32         | 97.18      | 97.12        |
| Quantized(Pruned version) | 96.56        | 96.79         | 96.56      | 96.49        |

### Performance Metrics (Inference Time, FPS, GMACs, Model Size)

| Model     | Avg Inference Time (ms/img) | Total Inference Time (ms) | FPS (Frames/sec) | GMACs (Giga MACs) | Model Size (MB) |
| :-------- | :-------------------------- | :------------------------ | :--------------- | :---------------- | :-------------- |
| Baseline(GPU)  | 0.9704                      | 12255.91                  | 1030.52          | 0.015422          | 1.5381          |
| Pruned(GPU)    | 0.9611                      | 12138.22                  | 1040.52          | 0.006685          | 0.6489          |
| Quantized(CPU) | 2.7998                      | 35360.98                  | 357.17           | 0.000081          | 0.1678          |

### Key Takeaways from Results:

  * **Inference Environment Note:** A key analysis point revealed that while reducing architecture size and parameters significantly, the observed inference time for the **quantized model (run on CPU)** was higher compared to the **baseline and pruned models (both run on GPU)**. This is primarily due to the overhead calculations inherent when running quantized models on CPU, which may not fully leverage the theoretical computational reductions.
  * **Superior Pruning:** Analysis-driven pruning maintained near-baseline accuracy, outperforming arbitrary fixed-percentage pruning.
  * **Accuracy Preservation:** Both pruning (97.18%) and QAT (96.56%) retained high accuracy, very close to the baseline (97.65%).
  * **Dramatic Size Reduction:** Pruning alone reduced size by \~60% (1.61 MB to 0.65 MB). Combined with QAT, size dropped by an impressive **\~90% (to 0.17 MB)**, making it ideal for highly constrained edge devices.
  * **Significant GMACs Reduction:** Pruning cut GMACs by over half. QAT further reduced them to an extremely low **0.000081 GMACs**, signifying massive gains in computational efficiency. This GMACs reduction is the core benefit for power consumption and throughput on dedicated integer hardware.

In essence, this project successfully optimized a CNN for traffic sign classification, achieving substantial reductions in size and computational cost with minimal impact on accuracy, demonstrating its viability for practical applications across various hardware platforms.

-----

## 📄 License

This project is licensed under the terms of the [LICENSE.txt](LICENSE.txt). See the file for details.


## 📧 Contact

Suraj Varma
