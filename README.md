# Time2Feat: Learning Interpretable Representations for Multivariate Time Series Clustering

Time2Feat is an end-to-end machine learning system for multivariate time series clustering. 
The system is the first to leverage
both inter-signal and intra-signal features of the time series. While
relying on state-of-the-art feature extraction approaches allows to
further refine the features by choosing the most appropriate ones
and incorporating human feedback in the feature selection process

## Installation

**time2feat** was tested on Python 3.7 on Linux amd Windows machines. It is recommended to use a virtual environment (
See: [python3 venv doc](https://docs.python.org/3/tutorial/venv.html)).

Clone the project into local and install **time2feat** package:

```bash
// Virtual environment creation
$ cd source

$ virtualenv -p python3 venv

$ source venv/bin/activate

// Dependecy installation
$ pip install -r requirements.txt
```

## Quick Start

Get started with **time2feat**

```python
import numpy as np
from t2f import feature_extraction
from t2f import feature_selection
from t2f import ClusterWrapper

# 10 multivariate time series with 100 timestamps and 3 signals each
arr = np.random.randn(10, 100, 3)

# labels = {} # unsupervised mode
labels = {0: 'a', 1: 'b'}  # semi-supervised mode

n_clusters = 2  # Number of clusters

# Feature extraction
df_feats = feature_extraction(arr, batch_size=100, p=1)

# Feature selection
top_feats = feature_selection(df_feats, labels=labels, auto=True)
df_feats = df_feats[top_feats]

# Clustering
model = ClusterWrapper(n_clusters=n_clusters, model_type='Hierarchical', transform_type='std')
y_pred = model.fit_predict(df_feats)
```

## Working example


- [Demo:](https://github.com/softlab-unimore/time2feat) A simple notebook to apply **time2feat** on UEA & UCR multivariate time series dataset.




## Dataset

All public multivariate time series datasets used in the paper can be downloaded from [UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/index.php).
The demo code only support sktime formatted ts files (like the sample of Cricket dataset).






