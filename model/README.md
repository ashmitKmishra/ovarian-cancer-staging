# Trained model

`best_ovarian_model.pth` (163 MB) is the trained network used for every result in this repository. It is larger than GitHub's 100 MB file limit, so it is attached to the repository's [Releases page](../../../releases) instead of being stored in git.

Download it and put it in this folder:

```
model/best_ovarian_model.pth
```

Then, from the repository root:

```bash
python code/evaluate.py
```

## What the file contains

A PyTorch `state_dict` (weights only, no code) for torchvision's ResNet-101 with its final layer replaced by `Dropout(0.3032)` followed by `Linear(2048, 5)`. `build_model()` in `code/common.py` creates the matching architecture. These are the weights from epoch 9 of the original final training run, which had the highest validation accuracy (97.52%).

SHA-256: `705b1abde812e859a6eea552830ee5d07836fe2a1107af0af639da23ff23347f`
