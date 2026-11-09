# Results

These files hold summary results only. `code/evaluate.py` also writes `test_predictions.csv` (the prediction for each test image, by filename), but that file is git-ignored because it lists image names.

| File | Made by | Contents |
|---|---|---|
| `test_results.txt` | `code/evaluate.py` | Test accuracy, per-class precision / recall / F1, confusion matrix |
| `confusion_matrix.csv` | `code/evaluate.py` | 5 x 5 counts; rows are the true class, columns the predicted class |
| `dataset_split.csv` | `code/figures/plot_dataset_split.py` | Image counts per class in the training, validation and test sets |
| `original_training_log.txt` | the original training run | Console output of the genetic algorithm search and final training |
| `original_ga_search_history.csv` | transcribed from the log | Validation loss and accuracy of each of the 17 candidates the search evaluated |
| `original_final_training_history.csv` | transcribed from the log | Training and validation loss and accuracy for each of the 14 epochs of the final run |

## About the original log

The original training script printed progress to the console and did not save metrics to a file. `original_training_log.txt` is that console output as it was copied at the time, minus three library deprecation warnings at the top. Because it was copied from a scrolling terminal, some lines appear more than once and some progress bars are cut off.

Values the log does not contain are left empty in the CSV files:

- validation loss and accuracy for epoch 1 and epoch 6 of the final run
- training loss and accuracy for epoch 3 of the final run

The genetic algorithm log does not record which hyperparameters each candidate used, only its score. It does record the selected hyperparameters ("Best hyperparameters"), which are stored in `code/common.py`.

Runs of the current `code/train.py` write `training_history.csv`, `ga_search_history.csv` (including each candidate's hyperparameters) and `hyperparameters.json` to `runs/latest/`.
