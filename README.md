# Drone RF Detection

This project explores the detection and classification of drones based on Radio Frequency (RF) signal analysis. It integrates statistical methods (Information Theory, Markov Chains) with classical Machine Learning techniques to distinguish between different drone devices.

## 🚀 Project Overview

The system analyzes I/Q (In-phase and Quadrature) signal features to:
1.  **Preprocess and Symbolize** signal data.
2.  **Model** signal transitions using Markov Chains.
3.  **Classify** drones using statistical entropy metrics and machine learning algorithms.

## 📂 Project Structure

- **`Statistical_Analysis_Codes/`**: The core statistical analysis pipeline.
    - `01_preprocessing_and_symbolization.ipynb`: Loads I/Q features, performs normalization, and utilizes K-Means clustering to convert continuous signals into discrete symbols.
    - `02_markov_modeling_and_entropy.ipynb`: constructs Markov Transition Matrices (MTM) from the symbolized data and calculates entropy rates.
    - `03_feature_selection_and_classification.ipynb`: Uses the derived statistical features for drone classification.

- **`Machine_Learning_Final_With_Results/`**:
    - `Classification_With_Normal_CLassical_ML(acc=97.7).ipynb`: A comprehensive notebook maximizing classification accuracy using classical ML models (e.g., Random Forest, XGBoost), achieving ~97.7% accuracy.

- **`Data_Extraction_For_DeepLearning_And_Classical_ML/`**:
    - `Data_Extraction_Code_Frome_DRONE_RF_SIGNAL.ipynb`: Scripts for extracting relevant features from raw Drone RF, enabling further Deep Learning and ML analysis.

- **`processed_data/`**: Directory for storing intermediate files generated during the analysis.
- **`results_Using_Pure_Statistics/`**: Stores results derived purely from the statistical methods.

## 🛠️ Requirements

To run this project, you need **Python 3.x** and **Jupyter Notebook**.

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## ▶️ Usage

1.  **Dataset**:
    Ensure the dataset file `mpact_iq_features_full_dataset.csv` is present in the project root or accessible.

2.  **Statistical Pipeline**:
    Navigate to `Statistical_Analysis_Codes` and run the notebooks in order:
    1.  `01_preprocessing_and_symbolization.ipynb`
    2.  `02_markov_modeling_and_entropy.ipynb`
    3.  `03_feature_selection_and_classification.ipynb`

    > **Note:** improved file paths! Check the `DATA_PATH` variable in the notebooks. You might need to update it to point to `../mpact_iq_features_full_dataset.csv` if running from the subdirectory.

3.  **Machine Learning Classification**:
    Run `Classification_With_Normal_CLassical_ML(acc=97.7).ipynb` in the `Machine_Learning_Final_With_Results` directory to see the high-accuracy classification results.

## 📊 Results

- **Statistical Approach**: Provides insights into the theoretical complexity and structure of the RF signals through entropy and transition probabilities.
- **ML Approach**: Demonstrates high practical accuracy (~97.7%) for drone identification.
