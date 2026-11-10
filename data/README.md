# Data

No data is stored in this repository: no images, image lists or labels. The microscopy images are available from the corresponding author, Dr. Prabhakar Pradhan (Department of Physics and Astronomy, Mississippi State University), on reasonable request.

## Setting up the data after you receive it

**1. Put the images in two folders:**

```
data/
└── images/
    ├── train_val/    2,418 images used for training and validation
    └── test/         126 held-out test images
```

**2. Create the two label files** from the image filenames:

```bash
python code/make_label_csv.py data/images/train_val data/train.csv
python code/make_label_csv.py data/images/test data/test.csv
```

The images are named by class, for example `Ov_C (12).jpg` or `Ov_III (4).jpg`, and the script turns the prefix into a label:

| Label | Class | Filename prefix | Development images | Test images |
|---:|---|---|---:|---:|
| 0 | Normal / benign control | `Ov_C` | 515 | 27 |
| 1 | Stage I | `Ov_I` | 1,174 | 61 |
| 2 | Stage II | `Ov_II` | 213 | 11 |
| 3 | Stage III | `Ov_III` | 366 | 19 |
| 4 | Stage IV | `Ov_IV` | 150 | 8 |

Check that the counts the script prints match this table. Each CSV has two columns, `image` (filename) and `label`. On the original images, the script reproduces the label files used in the study exactly.

`code/train.py` then splits `train.csv` 80/20 into training and validation sets, stratified by class, with `random_state=42`. `test.csv` is used only by `code/evaluate.py`. No image appears in both the development and test sets (checked by comparing file contents, not only names).
