# POint cloud LAtent Registration

# Getting started

POLAR is a method to simultaneously register numerous highly degraded point clouds corresponding to views of the same unknown reference object.

## Installation

> [!WARNING]
> PyTorch3D is required by POLAR. Since its installation is quite specific, you need to install it manually.

If you want to user POLAR with the provided pretrained autoencoder, run
```
pip install polaregistration
```

If you want to retrain the autoencoder on your own, run
```
pip install polaregistration[train]
```

If you want to use the interactive visualization function, run
```
pip install polaregistration[vis]
```

Finally, to install everything, run
```
pip install polaregistration[all]
```


## Minimal example

```python

from polar import load_sample_data, POLAR

X, degradations, R_abs_gt = load_sample_data()

model = POLAR(**degradations)
X_hat = model.fit_transform(X)
```
POLAR partially respects the [Scikit-Learn Estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator). Specifically, it has the two main methods:

- `fit(X: Tensor | Sequence[Tensor]) -> None` 
- `fit_transform(X: Tensor | Sequence[Tensor]) -> Tensor | Sequence[Tensor]`

They take a list of PyTorch tensors (each of shape $(K_i, 3)$ if point clouds are of varying lengths) or a single batch tensor of shape
$(N, K, 3)$ containing all the (same length) views.


## Documentation

A documentation, containing full api reference as well as small showcases is available [here](https://pypolar.github.io/polar/).
