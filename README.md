# Remote Sensing Image Segmentation

## Overview
This report contains implementation details and experimental results for remote sensing image segmentation using partial cross-entropy loss. The project explores the impact of hyperparameters on the performance of a segmentation network.

## 2. Methodology

### 2.1 Partial Cross-Entropy Loss

Partial cross-entropy loss is implemented to handle scenarios where only a subset of ground truth labels is available, suitable for remote sensing tasks where annotating every pixel is impractical.
```python
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PartialCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=1)
        loss = -torch.sum(target * log_prob) / input.size(0)
        return loss
```python

### 2.2 Remote Sensing Dataset and Point Label Simulation

A remote sensing image dataset is utilized, and point labels are simulated through random sampling.

### 2.3 Segmentation Network Architecture

A U-Net architecture is employed due to its effectiveness in image segmentation tasks, featuring an encoder-decoder structure. A U-Net architecture is employed due to its effectiveness in image segmentation tasks, featuring an encoder-decoder structure.


### 2.4 Training and Evaluation

The network is trained using the implemented partial cross-entropy loss, optimized with Adam and evaluated on a validation set.

## 3. Experimentation

### 3.1 Purpose

To investigate the impact of learning rate and batch size variations on the segmentation network's performance.

### 3.2 Hypothesis

Adjusting learning rates and batch sizes will affect training dynamics and accuracy, with lower rates potentially yielding better performance.

### 3.3 Experimental Process

Dataset Preparation: Prepare the remote sensing dataset and simulate point labels.
Network Training: Train the U-Net model using different combinations of learning rates (0.001, 0.01) and batch sizes (16, 32, 64).
Performance Evaluation: Evaluate each configuration based on validation loss and accuracy metrics.

### 3.4 Results

The following results were obtained from the experiments:

## Results

### Hyperparameter Optimization

| Learning Rate | Batch Size | Training Loss | Validation Loss | Validation Accuracy |
|---------------|------------|---------------|-----------------|---------------------|
| 0.001         | 16         | 0.8512        | 0.9334          | 67.53%              |
| 0.001         | 32         | 0.8494        | 0.9292          | 67.60%              |
| 0.001         | 64         | 0.8411        | 0.9355          | 67.33%              |
| 0.01          | 16         | 1.4638        | 1.5453          | 44.73%              |
| 0.01          | 32         | 2.3038        | 2.3037          | 10.00%              |
| 0.01          | 64         | 2.3041        | 2.3031          | 10.00%              |

### Example Results

![Sample Segmentation](results/sample_segmentation.png)

Include visualizations or example outputs from your segmentation models here.



## 4. Conclusion

The experiment demonstrates that a lower learning rate (0.001) generally leads to better performance in terms of validation accuracy. Larger batch sizes tend to slightly degrade performance, especially when coupled with higher learning rates. This suggests that careful tuning of hyperparameters is crucial for optimizing the segmentation network.

##5. Recommendations

Further investigation could include:

Fine-tuning hyperparameters more comprehensively, potentially exploring other optimization algorithms.
Exploring alternative network architectures or additional regularization techniques.
Scaling experiments with larger datasets and longer training durations to validate findings.
