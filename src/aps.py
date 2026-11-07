"""APS data preparation, logistic model trees, and evaluation."""
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

SEED = 552
ALPHAS = (0.0001, 0.001, 0.01)
def load_csv(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="latin1") as f:
        for i, line in enumerate(f):
            if line.startswith("class,"): break
        else: raise ValueError("Column header not found")
    frame = pd.read_csv(path, skiprows=i, na_values="na", encoding="latin1")
    y = frame.pop("class").map({"neg":0, "pos":1})
    assert y.notna().all()
    return frame.astype("float32"), y.to_numpy(dtype="int8")

def medians(x):
    return x.median().fillna(0)

def filled(x, med):
    return x.fillna(med).to_numpy(dtype=np.float32)

def transform(x):
    # Signed log compresses large counters and preserves zeros.
    return np.sign(x) * np.log1p(np.abs(x))

class LogisticTree:
    """One oblique split with an XGBoost L1 logistic score at each node."""
    def __init__(self, alpha): self.alpha = alpha
    def _fit_node(self, X, y):
        if np.unique(y).size == 1: return float(y[0])
        return XGBClassifier(booster="gblinear", objective="binary:logistic",
            reg_alpha=self.alpha, reg_lambda=0., n_estimators=20, learning_rate=0.08,
            updater="coord_descent", feature_selector="cyclic", n_jobs=1,
            random_state=SEED, verbosity=0).fit(X,y)
    @staticmethod
    def _score(model, X):
        if isinstance(model,float): return np.full(len(X),model)
        return model.predict_proba(X)[:,1]
    def fit(self, X, y):
        self.root = self._fit_node(X,y)
        p = self._score(self.root,X)
        stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=100,
                                      random_state=SEED).fit(p[:,None],y)
        self.cut = None
        if stump.tree_.node_count > 1:
            self.cut = float(stump.tree_.threshold[0])
            left = p <= self.cut
            self.leaves = [self._fit_node(X[left],y[left]), self._fit_node(X[~left],y[~left])]
        return self
    def prob(self, X):
        p = self._score(self.root,X)
        if self.cut is None: return p
        left = p <= self.cut
        out = np.empty(len(X))
        for mask, node in zip((left,~left),self.leaves):
            if np.any(mask): out[mask] = self._score(node,X[mask])
        return out

def prep(A,B,mb):
    ma = medians(A)
    xa, xb = transform(filled(A,ma)), transform(filled(B,mb))
    scale = StandardScaler().fit(xa)
    return scale.transform(xa), scale.transform(xb)

def tune(A, y):
    records=[]
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(A,y),1):
        # Learn fold medians and scaling before generating any synthetic data.
        xa,xb=prep(A.iloc[a], A.iloc[b], medians(A.iloc[a]))
        for use in (False,True):
            xt,yt = SMOTE(random_state=SEED+fold).fit_resample(xa,y[a]) if use else (xa,y[a])
            for alpha in ALPHAS:
                model=LogisticTree(alpha).fit(xt,yt)
                p=model.prob(xb)
                records.append(dict(fold=fold,smote=use,alpha=alpha,
                    n_validation=len(b),n_fit=len(yt),error=np.mean((p>=.5)!=y[b]),
                    auc=roc_auc_score(y[b],p)))
    return pd.DataFrame(records)

def evaluate(name,split,y,p,oob=np.nan):
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(model=name,split=split,TN=tn,FP=fp,FN=fn,TP=tp,
        error=(fp+fn)/len(y),auc=roc_auc_score(y,p),recall=tp/(tp+fn),
        precision=tp/(tp+fp) if tp+fp else 0.,oob_error=oob)
