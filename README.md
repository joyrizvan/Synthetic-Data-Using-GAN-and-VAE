# GAN-vs-VAE-synthetic-data-comparison

### Overview
This project explores the use of generative AI models—specifically CTGAN and TVAE—to generate synthetic tabular data for improving customer churn prediction. Traditional churn datasets often suffer from class imbalance (few churners vs. many non-churners) and privacy constraints that limit data access. Synthetic data offers a promising solution to both challenges.

### Objective
Use synthetic data to balance the training dataset for churn prediction.

Evaluate whether models trained on synthetic or augmented datasets perform better than those trained on real or oversampled data.

Compare generative models: CTGAN vs. TVAE.

### Methodology
Dataset: Real-world customer churn dataset (with features like age, tenure, support calls, etc.).

Models used:

Synthetic data generation: CTGAN, TVAE (via SDV)

Classification: Logistic Regression, Random Forest

Evaluation Metrics: Accuracy, F1-score, AUC-ROC

### Key Findings
CTGAN outperforms TVAE in generating statistically similar and useful synthetic data.

Models trained on synthetic + real (augmented) data achieve better performance than real-only or duplicated data.

CTGAN-generated synthetic data alone can outperform real data in some cases.

### Project Structure
<pre><code>```bash ├── data/ │ ├── original/ # Real training dataset │ ├── synthetic/ # Generated synthetic data ├── model/ # Saved .pkl models ├── services/ │ ├── preprocess.py # Preprocessing and transformation logic │ ├── churn_model.py # Training logic for ML models │ ├── evaluation.py # Model evaluation functions ├── notebooks/ # Jupyter notebooks for EDA & results ├── data_generator.py # Script to generate balanced synthetic data ├── requirements.txt # List of dependencies └── README.md # Project overview ``` </code></pre>
### How to Run
1. Install dependencies

bash
Copy
Edit
pip install -r requirements.txt

2. Train model with different scenarios
Run training scripts (churn_model.py or data_generator.py) to train and evaluate under:
Real-only
Synthetic-only
Duplicated
Augmented (real + synthetic)

3. Generate synthetic data
bash
Copy
Edit
python data_generator.py

### Requirements
Python 3.9+
SDV (ctgan, tvae)
scikit-learn
pandas, numpy, seaborn, matplotlib
