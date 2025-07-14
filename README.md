# leaf-disease-detector
Repository Name Recommendation: leaf-disease-detector or plant-disease-classification
# AI-Powered Leaf Disease Detection System

![Sample Prediction](images/sample_prediction.png) ## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Objectives](#2-objectives)
- [3. Dataset](#3-dataset)
- [4. Project Structure](#4-project-structure)
- [5. Setup and Installation](#5-setup-and-installation)
- [6. Usage](#6-usage)
- [7. Model Architecture](#7-model-architecture)
- [8. Results](#8-results)
- [9. Explainable AI (XAI)](#9-explainable-ai-xai)
- [10. Contributing](#10-contributing)
- [11. License](#11-license)
- [12. Contact](#12-contact)

---

## 1. Introduction

Plant diseases pose a significant threat to global food security, leading to substantial crop losses and economic hardship for farmers. Traditional methods of disease detection often rely on manual inspection, which is time-consuming, prone to human error, and requires expert knowledge. This project aims to address these challenges by leveraging the power of Artificial Intelligence (AI) and Machine Learning (ML), specifically Deep Learning using Convolutional Neural Networks (CNNs), to develop an automated, accurate, and efficient system for detecting and classifying plant leaf diseases from images.

The project is designed to be easily reproducible and shareable, using a structured repository layout and developed within an interactive Jupyter Notebook environment.

## 2. Objectives

* To develop a robust CNN model capable of accurately identifying and classifying various plant leaf diseases.
* To create a system that can distinguish between healthy and diseased leaves across multiple plant species.
* To evaluate the performance of transfer learning using a pre-trained CNN architecture (e.g., MobileNetV2) on the PlantVillage dataset.
* To provide a clear, reproducible, and well-documented project that can be easily set up and run by others.
* To demonstrate the full machine learning pipeline: data loading, preprocessing, model training, evaluation, and prediction within a Jupyter Notebook.
* To offer basic model interpretability using Grad-CAM to understand model decisions.

## 3. Dataset

This project utilizes the **PlantVillage Dataset**, a widely recognized dataset for plant disease detection. It contains a large collection of healthy and diseased plant leaf images across various species.

**Due to its size, the dataset itself is NOT included in this repository.** You will need to download and extract it manually.

**Download Instructions:**
1.  Download the `PlantVillage.zip` dataset from either:
    * **Kaggle:** [PlantVillage Dataset](https://www.kaggle.com/datasets/saroz16/plantvillage-dataset)
    * **Mendeley Data:** [PlantVillage Dataset (Original Source)](https://data.mendeley.com/datasets/tywbtsjrjv/1)
2.  Once downloaded, create a directory named `data` in the root of this cloned repository.
3.  Extract the contents of the `PlantVillage.zip` file. Ensure that the extracted folder structure places the `train` and `validation` (or `test` if present) subdirectories directly inside `leaf-disease-detection-ai/data/PlantVillage/`.

    **Expected Directory Structure After Extraction:**
    ```
    leaf-disease-detection-ai/
    ├── data/
    │   └── PlantVillage/
    │       ├── train/
    │       │   ├── Apple___Apple_scab/
    │       │   ├── Apple___Black_rot/
    │       │   └── ... (other disease folders)
    │       └── validation/
    │           ├── Apple___Apple_scab/
    │           ├── Apple___Black_rot/
    │           └── ...
    └── ... (other project files)
    ```

## 4. Project Structure
leaf-disease-detection-ai/
├── .gitignore             # Files/folders to ignore in Git
├── README.md              # Project overview, setup, usage
├── LICENSE                # Open-source license
├── requirements.txt       # Python dependencies
├── data/                  # Placeholder for the dataset (PlantVillage will be here)
│   └── .gitkeep           # Allows 'data' folder to be tracked when empty
├── notebooks/             # Jupyter notebook for project development
│   └── leaf_disease_detection.ipynb  # Main project notebook
├── trained_models/        # Saved model weights
│   └── .gitkeep           # Allows 'trained_models' folder to be tracked when empty
├── images/                # Sample prediction images, plots for README.md
│   └── sample_prediction.png
└── src/                   # Optional: Python scripts for reusable functions

## 5. Setup and Installation

Follow these steps to set up the project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/leaf-disease-detection-ai.git](https://github.com/YourGitHubUsername/leaf-disease-detection-ai.git)
    cd leaf-disease-detection-ai
    ```
2.  **Create and activate a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    * First, you'll need to run the notebook once to generate `requirements.txt` with exact versions. For initial setup, you might need to manually install `tensorflow` and `jupyter` first.
    * Then, install all required libraries:
        ```bash
        pip install -r requirements.txt
        ```
    * *(If `requirements.txt` is not yet present or you're setting up for the first time before running the notebook):*
        ```bash
        pip install jupyter tensorflow matplotlib seaborn scikit-learn pillow opencv-python tf-keras-vis
        ```
4.  **Download and extract the dataset** as described in the [Dataset section](#3-dataset).

## 6. Usage

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:** In the Jupyter interface that opens in your browser, navigate to the `notebooks/` directory and click on `leaf_disease_detection.ipynb`.
3.  **Run Cells:** Execute each cell sequentially from top to bottom (`Shift + Enter`).
    * The notebook will guide you through data loading, preprocessing, model definition, training, evaluation, and making predictions.
    * It will automatically save the best-trained model to the `trained_models/` directory.

## 7. Model Architecture

The project utilizes a deep learning approach based on Convolutional Neural Networks (CNNs). Specifically, it employs **Transfer Learning** with a pre-trained **MobileNetV2** model.

* **Base Model:** MobileNetV2, pre-trained on the ImageNet dataset. The top classification layer is removed.
* **Frozen Layers:** The weights of the MobileNetV2 base model are frozen during initial training to leverage learned features.
* **Custom Head:** A custom classification head is added on top, consisting of:
    * `GlobalAveragePooling2D`: To reduce feature maps to a single feature vector per channel.
    * `Dense` layer with ReLU activation.
    * `Dropout`: For regularization to prevent overfitting.
    * Final `Dense` layer with `softmax` activation for multi-class classification.
* **Optimizer:** Adam optimizer with a low learning rate (e.g., 0.0001) for fine-tuning.
* **Loss Function:** Categorical Cross-Entropy.

## 8. Results

The model's performance will be evaluated on the validation set using metrics such as:

* **Accuracy:** Overall correct predictions.
* **Precision, Recall, F1-Score:** Per-class performance metrics.
* **Confusion Matrix:** A visual representation of classification performance.

*(Once you've run the notebook, you can add specific metrics here, e.g.:)*

> After training for X epochs, the model achieved approximately **YY.YY% Validation Accuracy**.
>
> *(You can also add images of your training/validation plots and confusion matrix here, saving them to `images/` and referencing them like `![Confusion Matrix](images/confusion_matrix.png)`)*

## 9. Explainable AI (XAI)

To provide insights into why the model makes certain predictions, the project includes an optional section demonstrating **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM generates a heatmap highlighting the regions of the input image that were most important for the model's prediction, helping to visualize what features the CNN focused on.

## 10. Contributing

Contributions are welcome! If you have suggestions for improvements, find bugs, or want to add new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.
