# ACDistill(ACB and Distillation)

This demo will show you how to
1. Build an ACNet using Asymmetric Convolution Blocks
2. Train ACNet side-by-side with a standard CNN baseline, using identical training configurations for a fair comparison.
3. Evaluate both models to obtain and compare their average test accuracies.
4. Convert the trained ACNet into a model with the same structure as the baseline CNN for deployment.
5. Implement Knowledge Distillation (KD) to transfer knowledge from the trained ACNet (teacher) to a smaller student model for efficient inference.

Quick Start:
1. We use colab, single card with L4 GPU 
2. !git clone https://github.com/chedana/ADLS.git
3. !pip install  coloredlogs
4. Train: !python acnet/do_acnet.py --config config.yaml
5. Test : !python acnet/do_acnet.py --config config.yaml -e True
Some results (Top-1 accuracy) reproduced on CIFAR-10 using the codes in this repository:

| Model       | Baseline | ACNet |
|-------------|----------|-------|
| Cifar-quick | 84.64    | 85.75 |
| VGG         | 91.70    | 92.82 |
| Lenet       | 83.72    | 85.30 |



| Model        | Baseline | KD_logits | KD_feature |
|--------------|----------|-----------|------------|
| Cifar-quick  | 84.64    | 85.61     | 85.51      |
| VGG_shallow  | 90.19    | 91.12     | 90.22      |
| Lenet        | 83.72    | 85.98     | 86.15      |

All Training Config.yaml, Training logs and Checkpoints can be found at [here](https://drive.google.com/drive/folders/174RHIPqfWNLO3g_DFQSpD7_yinjfGhnZ)

This work is based on [DingXiaoH/ACNet](https://github.com/DingXiaoH/ACNet).  
We sincerely thank the original authors for their excellent implementation.
