# 🧠 BIOPAC Cardiorespiratory ML Pipeline

This repository implements an advanced machine learning pipeline to classify physical activity levels based on ECG and pulmonary signals collected from BIOPAC systems.

The dataset is derived from 7 years of lab practicals at the University of Texas at Arlington (UTA) involving students enrolled in Biomedical Engineering courses.

---

## 📊 Project Overview

The goal is to classify subjects into four activity categories:

- `athlete`
- `regular`
- `mixed`
- `sedentary`

using 30+ physiological features extracted from BIOPAC recordings. These include heart rate variability, tidal volume, inspiratory reserve, expiratory volumes, and other cardiorespiratory metrics.

---

## 🧱 Project Structure

biopac-cardiorespiratory-ml/
├── data/ # Input CSV files
├── notebooks/ # Optional Jupyter explorations
├── biopac_pipeline/ # Modular pipeline code
│ ├── init.py
│ ├── data_loader.py # Data loading
│ ├── preprocess.py # Encoding, scaling, filtering
│ ├── model.py # ML pipeline definition
│ └── evaluate.py # CV + reporting
├── main.py # Run the full pipeline
├── requirements.txt
├── README.md



---

## ⚙️ How to Run

### 1. Clone the repository

git clone https://github.com/SubratBastola/biopac-cardiorespiratory-ml.git
cd biopac-cardiorespiratory-ml
2. Install dependencies
pip install -r requirements.txt
3. Add your dataset
Place your CSV (e.g., class_data_2022_2023_2024.csv) in the data/ folder.

4. Run the full ML pipeline

python main.py
This will print per-fold classification reports and ROC-AUC scores, and visualize confusion matrices and feature importances.

🧠 Methods
Preprocessing:

Variance thresholding

Label encoding

Standard scaling

Feature selection with ANOVA F-test

Model:

Random Forest (base)

Logistic Regression (meta)

StackingClassifier from scikit-learn

Evaluation:

Stratified 5-fold CV

Multi-class ROC-AUC

Confusion matrix + classification report

🏆 Results
Typical performance (with 30% noise + 15% label flip):

Mean ROC-AUC ≈ 0.78–0.83

Precision & recall vary across activity types, indicating real-world variability

📌 Citation
If using this code or adapted dataset, please cite:

Bastola, S. (2025). BIOPAC Cardiorespiratory ML Pipeline (v1.0) [Computer software]. GitHub. https://github.com/SubratBastola/biopac-cardiorespiratory-ml

📜 License
MIT License. Free to use and modify with attribution.

🙌 Acknowledgments
UTA Biomedical Engineering Lab Instructors

Students from 2017–2024 cohort

BIOPAC Systems Inc.
