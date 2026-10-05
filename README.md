# Breast Cancer Diagnosis: Neural Network vs. Classical ML, with SHAP Explainability

Binary classification of breast tumours (malignant vs. benign) from cell-nucleus measurements, comparing a small neural network against Logistic Regression, Random Forest and SVM, and using SHAP to explain the predictions.

> \\\*\\\*Status:\\\*\\\* Course/research exercise. Not a clinical tool.

## Dataset

[Wisconsin Diagnostic Breast Cancer (WDBC)](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) (Wolberg, Street \& Mangasarian, UCI Machine Learning Repository).

* 569 samples, 30 numeric features (mean, standard error and "worst" value of 10 nuclear measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
* Classes: 357 benign (62.7%), 212 malignant (37.3%)
* No missing values, no duplicate rows. The only cleanup needed was dropping an empty trailing column (`Unnamed: 32`) and the `id` column.

## Approach

1. Load and inspect the data; encode `M`→1, `B`→0
2. Correlation analysis (mean features)
3. 80/20 train/test split (`random\\\_state=42`), 455 train / 114 test
4. Standardise features (scaler fit on the training set only)
5. Train four models: Logistic Regression, Random Forest, SVM (RBF, probability enabled) and a small neural network (30 → Dense(16, ReLU) → Dropout(0.2) → Dense(1, sigmoid), 513 parameters, Adam, 100 epochs)
6. Compare on accuracy, precision, recall, F1 and ROC AUC
7. Explain the neural network and the SVM with SHAP

## Results (held-out test set, n = 114)

|Model|Accuracy|Precision|Recall|F1|ROC AUC|
|-|-|-|-|-|-|
|Logistic Regression|97.37%|97.62%|95.35%|96.47%|0.9974|
|Random Forest|96.49%|97.56%|93.02%|95.24%|0.9953|
|SVM|98.25%|100.00%|95.35%|97.62%|0.9974|
|Neural Network|98.25%|97.67%|97.67%|97.67%|0.9964|

!\[ROC curves](figures/roc\_deep\_learn\_vs\_ML.png)

**How to read this:** with 114 test samples, one extra misclassification changes accuracy by about 0.9 percentage points. The gaps between models here are 1–2 samples, so these results do not establish that any model is better than another. The neural network has the highest recall (fewest missed malignant cases), which matters most in a screening setting, but this too is a one-or-two-sample difference.

## Explainability

SHAP summary plots for  SVM:
!\[SHAP SVM](figures/shap\_sv.png)

## Limitations

* Single train/test split on a small dataset; no cross-validation or confidence intervals
* The neural network's "validation" curves are computed on the test set, so the test set is not fully unseen for that model
* No hyperparameter tuning

## Reproduce

```bash
git clone https://github.com/Abdulwaliyi007/breast-cancer-nn-vs-ml.git
cd breast-cancer-nn-vs-ml
pip install -r requirements.txt
jupyter notebook breast\\\_cancer\\\_dl.ipynb
```

`data.csv` must sit in the same folder as the notebook.

## Repository structure

```
├── breast\\\_cancer\\\_dl.ipynb
├── data.csv
├── requirements.txt
├── figures/
└── README.md
```

## Citation and data licence

Dataset: W.H. Wolberg, W.N. Street, O.L. Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository. Check the licence on the UCI page and include it here.

## Disclaimer

For educational and research purposes only. Not validated for clinical use.

