# Early Stopping

This project supports early stopping during training to prevent overfitting and save computational resources.

## How It Works

Early stopping monitors a specified metric on the target domain data during training. If the metric doesn't improve for a specified number of epochs (patience), training stops early.

## Configuration

Add these parameters to your config file under the `training` section:

```yaml
training:
  # ... other training parameters ...
  
  # Early stopping configuration
  early_stopping_patience: 10  # Number of epochs to wait for improvement
  early_stopping_metric: "f1"  # Metric to monitor: 'f1' or 'accuracy'
```

### Parameters

- **`early_stopping_patience`**: Number of consecutive epochs without improvement before stopping
  - Set to an integer (e.g., `10`) to enable early stopping
  - Set to `null` or omit to disable early stopping
  - Recommended: 5-15 epochs depending on your training dynamics

- **`early_stopping_metric`**: Metric to monitor on target data
  - `"f1"`: Macro F1-score (recommended for imbalanced datasets)
  - `"accuracy"`: Classification accuracy
  - The metric is evaluated on the target domain data after each epoch

## Example Configurations

### Example 1: F1-based Early Stopping
```yaml
training:
  method: "baseline"
  num_epochs: 100
  lr: 0.001
  early_stopping_patience: 10
  early_stopping_metric: "f1"
```

### Example 2: Accuracy-based Early Stopping
```yaml
training:
  method: "sinkhorn"
  num_epochs: 50
  lr: 0.0001
  early_stopping_patience: 15
  early_stopping_metric: "train_loss"
```

### Example 3: Disabled Early Stopping
```yaml
training:
  method: "baseline"
  num_epochs: 50
  lr: 0.001
  early_stopping_patience: null  # Disabled
```

## Training Output

When early stopping is enabled, you'll see additional logging during training:

```
✓ Early stopping enabled: metric=f1, patience=10

Epoch 1/100: ...
Target F1: 0.7234

Epoch 2/100: ...
Target F1: 0.7456

...

Epoch 15/100: ...
Target F1: 0.8123
EarlyStopping counter: 1 out of 10 (best f1: 0.8245)

...

Epoch 25/100: ...
Target F1: 0.8156
EarlyStopping counter: 10 out of 10 (best f1: 0.8245)
Early stopping triggered! No improvement in f1 for 10 epochs.
Stopping training at epoch 25
```

## Notes

- Early stopping only works when a target data loader is provided during training
- For baseline (source-only) training with target data, early stopping helps find the optimal stopping point
- The best model (based on training loss) is still saved and loaded at the end of training
- Early stopping is evaluated after each full epoch

## Implementation Details

The early stopping mechanism:
1. Evaluates the specified metric on the entire target dataset after each epoch
2. Tracks the best metric value seen so far
3. Maintains a counter of consecutive epochs without improvement
4. Stops training when the counter reaches the patience threshold
5. Uses the model state at the best metric value (via best_model tracking)

