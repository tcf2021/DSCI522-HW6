# DSCI 552 — Homework 6

Chengfeng Tang · GitHub: tcf2021

[Homework notebook](notebook/Tang_Chengfeng_HW6.ipynb) contains the executed APS failure classification experiments, answers to the ISLR exercises, and extra practice. The confusion matrices use actual classes as rows and predicted classes as columns; positive means an APS-related failure.

| Path | Contents |
|---|---|
| `notebook/` | Executed notebook with answers, figures, and tables |
| `data/aps/` | Official train and test CSVs (losslessly gzip-compressed), with the original dataset description |
| `src/aps.py` | Data loading, fold preprocessing, logistic model tree, and evaluation |
| `results/` | CVs for all 170 features, correlation matrix, fold-level validation results, classification metrics, and figures |

## Running the notebook

Tested with Python 3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m notebook notebook/Tang_Chengfeng_HW6.ipynb
```

Run the cells in order. The data reader opens the compressed CSVs directly; no download or manual extraction is needed. The supplied 60,000 training and 16,000 test observations stay in their original splits. Each SMOTE fit uses only the training portion of its fold. The final fit uses all 59,000 negatives and balances them with 59,000 positives. The ROC curves and confusion matrices always use original observations.

Data: [APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), Scania CV AB. The original notices are retained in the source files.
