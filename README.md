# CIFAR10 Image Classification with Custom CNN

This project implements a custom CNN architecture for CIFAR10 image classification with specific architectural requirements including depthwise separable convolution, dilated convolution, and a receptive field > 44.

## Model Architecture

The model uses a custom architecture with the following key components:

1. **Input Block**: 
   - Conv 3x3 (3→24 channels)
   - Output: 32x32x24, RF=3

2. **Convolution Block 1**: 
   - Depthwise Separable Conv 3x3 (24→48 channels)
   - Output: 16x16x48, RF=5

3. **Convolution Block 2**:
   - First Conv: Dilated Conv 3x3 (48→64 channels)
   - Second Conv: Standard Conv 3x3 (64→96 channels)
   - Output: 8x8x96, RF=13

4. **Convolution Block 3**:
   - Conv 3x3 (96→128 channels)
   - Output: 4x4x128, RF=21

5. **Output Block**:
   - Global Average Pooling
   - 1x1 Convolution to get 10 classes
   - Final RF=45

### Meeting Architecture Requirements

1. ✅ Receptive Field > 44 (Achieved: 45)
2. ✅ Used Depthwise Separable Convolution in Block 1
3. ✅ Used Dilated Convolution in Block 2
4. ✅ Used GAP layer
5. ✅ Parameters < 200k (Achieved: 197,922)

## Training Features

- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01 with StepLR scheduler
- Batch Size: 64
- Dropout: 0.02
- Loss Function: Negative Log Likelihood

## Model Training Instructions

1. Install requirements:
```bash
pip install torch torchvision tqdm matplotlib
```

2. Run training:
```bash
python src/main.py
```


## Training Summary

The model training includes:
- 20 epochs
- Learning rate scheduling (step size=6, gamma=0.1)
- Progress bars with live metrics
- Loss and accuracy plotting

## Best Accuracy Score

- Training Accuracy: 82.20%
- Test Accuracy: 85.46%

## Project Structure
```
├── src/
│ ├── model.py # Model architecture
│ ├── train.py # Training functions
│ ├── validate.py # Validation functions
│ ├── dataloader.py # Data loading utilities
│ └── main.py # Training script
├── README.md
└── training.log
```

## Model Parameters Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 32, 32]             648
       BatchNorm2d-2           [-1, 24, 32, 32]              48
              ReLU-3           [-1, 24, 32, 32]               0
           Dropout-4           [-1, 24, 32, 32]               0
            Conv2d-5           [-1, 24, 16, 16]             240
            Conv2d-6           [-1, 48, 16, 16]           1,200
DepthwiseSeparableConv-7           [-1, 48, 16, 16]               0
       BatchNorm2d-8           [-1, 48, 16, 16]              96
              ReLU-9           [-1, 48, 16, 16]               0
          Dropout-10           [-1, 48, 16, 16]               0
           Conv2d-11           [-1, 64, 16, 16]          27,712
      BatchNorm2d-12           [-1, 64, 16, 16]             128
             ReLU-13           [-1, 64, 16, 16]               0
          Dropout-14           [-1, 64, 16, 16]               0
           Conv2d-15             [-1, 96, 8, 8]          55,392
      BatchNorm2d-16             [-1, 96, 8, 8]             192
             ReLU-17             [-1, 96, 8, 8]               0
          Dropout-18             [-1, 96, 8, 8]               0
           Conv2d-19            [-1, 128, 4, 4]         110,720
      BatchNorm2d-20            [-1, 128, 4, 4]             256
             ReLU-21            [-1, 128, 4, 4]               0
          Dropout-22            [-1, 128, 4, 4]               0
AdaptiveAvgPool2d-23            [-1, 128, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]           1,290
================================================================
Total params: 197,922
Trainable params: 197,922
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.02
Params size (MB): 0.76
Estimated Total Size (MB): 2.78
----------------------------------------------------------------
```

## Logs
```
EPOCH:0 Loss=1.6572844982147217 Batch_id=781 Accuracy=39.12 LR=0.01
Test set: Average loss: 1.3822, Accuracy: 4965/10000 (49.65%)

EPOCH:1 Loss=1.0867810249328613 Batch_id=781 Accuracy=50.20 LR=0.01
Test set: Average loss: 1.1864, Accuracy: 5824/10000 (58.24%)

EPOCH:2 Loss=1.3571726083755493 Batch_id=781 Accuracy=55.15 LR=0.01
Test set: Average loss: 1.0707, Accuracy: 6203/10000 (62.03%)

EPOCH:3 Loss=1.4282718896865845 Batch_id=781 Accuracy=58.15 LR=0.01
Test set: Average loss: 1.0491, Accuracy: 6332/10000 (63.32%)

EPOCH:4 Loss=0.8133314847946167 Batch_id=781 Accuracy=60.82 LR=0.01
Test set: Average loss: 0.9282, Accuracy: 6665/10000 (66.65%)

EPOCH:5 Loss=0.8087322115898132 Batch_id=781 Accuracy=62.30 LR=0.01
Test set: Average loss: 0.9116, Accuracy: 6789/10000 (67.89%)

EPOCH:6 Loss=1.1159751415252686 Batch_id=781 Accuracy=63.63 LR=0.01
Test set: Average loss: 0.8536, Accuracy: 7025/10000 (70.25%)

EPOCH:7 Loss=1.4197944402694702 Batch_id=781 Accuracy=65.06 LR=0.01
Test set: Average loss: 0.8237, Accuracy: 7123/10000 (71.23%)

EPOCH:8 Loss=0.9810140132904053 Batch_id=781 Accuracy=66.34 LR=0.01
Test set: Average loss: 0.8079, Accuracy: 7184/10000 (71.84%)

EPOCH:9 Loss=0.7037351131439209 Batch_id=781 Accuracy=67.04 LR=0.01
Test set: Average loss: 0.8543, Accuracy: 7054/10000 (70.54%)

EPOCH:10 Loss=0.7341941595077515 Batch_id=781 Accuracy=68.05 LR=0.01
Test set: Average loss: 0.7629, Accuracy: 7371/10000 (73.71%)

EPOCH:11 Loss=0.8248751759529114 Batch_id=781 Accuracy=68.51 LR=0.01
Test set: Average loss: 0.7394, Accuracy: 7443/10000 (74.43%)

EPOCH:12 Loss=1.1625503301620483 Batch_id=781 Accuracy=69.21 LR=0.01
Test set: Average loss: 0.7343, Accuracy: 7431/10000 (74.31%)

EPOCH:13 Loss=1.0118250846862793 Batch_id=781 Accuracy=69.82 LR=0.01
Test set: Average loss: 0.6819, Accuracy: 7665/10000 (76.65%)

EPOCH:14 Loss=0.7851751446723938 Batch_id=781 Accuracy=70.23 LR=0.01
Test set: Average loss: 0.6676, Accuracy: 7680/10000 (76.80%)

EPOCH:15 Loss=0.9510664343833923 Batch_id=781 Accuracy=70.73 LR=0.01
Test set: Average loss: 0.7026, Accuracy: 7597/10000 (75.97%)

EPOCH:16 Loss=1.3664476871490479 Batch_id=781 Accuracy=71.53 LR=0.01
Test set: Average loss: 0.6406, Accuracy: 7774/10000 (77.74%)

EPOCH:17 Loss=0.6593987345695496 Batch_id=781 Accuracy=72.03 LR=0.01
Test set: Average loss: 0.6339, Accuracy: 7782/10000 (77.82%)

EPOCH:18 Loss=0.5893585681915283 Batch_id=781 Accuracy=72.40 LR=0.01
Test set: Average loss: 0.6333, Accuracy: 7798/10000 (77.98%)

EPOCH:19 Loss=0.6469125747680664 Batch_id=781 Accuracy=73.00 LR=0.01
Test set: Average loss: 0.6376, Accuracy: 7794/10000 (77.94%)

EPOCH:20 Loss=0.9000063538551331 Batch_id=781 Accuracy=73.39 LR=0.01
Test set: Average loss: 0.6372, Accuracy: 7829/10000 (78.29%)

EPOCH:21 Loss=0.5601007342338562 Batch_id=781 Accuracy=73.39 LR=0.01
Test set: Average loss: 0.6285, Accuracy: 7827/10000 (78.27%)

EPOCH:22 Loss=0.8293447494506836 Batch_id=781 Accuracy=73.71 LR=0.01
Test set: Average loss: 0.6114, Accuracy: 7896/10000 (78.96%)

EPOCH:23 Loss=1.1598211526870728 Batch_id=781 Accuracy=73.87 LR=0.01
Test set: Average loss: 0.6280, Accuracy: 7832/10000 (78.32%)

EPOCH:24 Loss=0.7866760492324829 Batch_id=781 Accuracy=74.21 LR=0.01
Test set: Average loss: 0.6084, Accuracy: 7940/10000 (79.40%)

EPOCH:25 Loss=0.8429051041603088 Batch_id=781 Accuracy=74.84 LR=0.01
Test set: Average loss: 0.5893, Accuracy: 8011/10000 (80.11%)

EPOCH:26 Loss=0.6797723770141602 Batch_id=781 Accuracy=74.84 LR=0.01
Test set: Average loss: 0.5849, Accuracy: 7978/10000 (79.78%)

EPOCH:27 Loss=0.6639786958694458 Batch_id=781 Accuracy=74.98 LR=0.01
Test set: Average loss: 0.5722, Accuracy: 8032/10000 (80.32%)

EPOCH:28 Loss=0.5467668771743774 Batch_id=781 Accuracy=75.41 LR=0.01
Test set: Average loss: 0.5610, Accuracy: 8104/10000 (81.04%)

EPOCH:29 Loss=0.8408741354942322 Batch_id=781 Accuracy=75.64 LR=0.01
Test set: Average loss: 0.5656, Accuracy: 8037/10000 (80.37%)

EPOCH:30 Loss=0.6513899564743042 Batch_id=781 Accuracy=75.89 LR=0.01
Test set: Average loss: 0.5652, Accuracy: 8103/10000 (81.03%)

EPOCH:31 Loss=0.5639716386795044 Batch_id=781 Accuracy=76.30 LR=0.01
Test set: Average loss: 0.5752, Accuracy: 8058/10000 (80.58%)

EPOCH:32 Loss=0.30553925037384033 Batch_id=781 Accuracy=76.12 LR=0.01
Test set: Average loss: 0.5486, Accuracy: 8141/10000 (81.41%)

EPOCH:33 Loss=1.0303934812545776 Batch_id=781 Accuracy=76.22 LR=0.01
Test set: Average loss: 0.5446, Accuracy: 8142/10000 (81.42%)

EPOCH:34 Loss=0.40041908621788025 Batch_id=781 Accuracy=76.76 LR=0.01
Test set: Average loss: 0.5371, Accuracy: 8183/10000 (81.83%)

EPOCH:35 Loss=0.4932129979133606 Batch_id=781 Accuracy=76.57 LR=0.01
Test set: Average loss: 0.5429, Accuracy: 8161/10000 (81.61%)

EPOCH:36 Loss=0.6026492714881897 Batch_id=781 Accuracy=77.17 LR=0.01
Test set: Average loss: 0.5347, Accuracy: 8177/10000 (81.77%)

EPOCH:37 Loss=0.6226880550384521 Batch_id=781 Accuracy=77.45 LR=0.01
Test set: Average loss: 0.5190, Accuracy: 8218/10000 (82.18%)

EPOCH:38 Loss=0.5824570655822754 Batch_id=781 Accuracy=77.31 LR=0.01
Test set: Average loss: 0.5536, Accuracy: 8112/10000 (81.12%)

EPOCH:39 Loss=0.8392065763473511 Batch_id=781 Accuracy=77.74 LR=0.01
Test set: Average loss: 0.5542, Accuracy: 8126/10000 (81.26%)

EPOCH:40 Loss=0.8491493463516235 Batch_id=781 Accuracy=77.36 LR=0.01
Test set: Average loss: 0.5192, Accuracy: 8256/10000 (82.56%)

EPOCH:41 Loss=0.7517022490501404 Batch_id=781 Accuracy=77.67 LR=0.01
Test set: Average loss: 0.5203, Accuracy: 8215/10000 (82.15%)

EPOCH:42 Loss=0.42158812284469604 Batch_id=781 Accuracy=78.11 LR=0.01
Test set: Average loss: 0.5324, Accuracy: 8225/10000 (82.25%)

EPOCH:43 Loss=0.7770720720291138 Batch_id=781 Accuracy=78.06 LR=0.01
Test set: Average loss: 0.5164, Accuracy: 8229/10000 (82.29%)

EPOCH:44 Loss=0.9110986590385437 Batch_id=781 Accuracy=78.24 LR=0.01
Test set: Average loss: 0.5282, Accuracy: 8209/10000 (82.09%)

EPOCH:45 Loss=0.615362286567688 Batch_id=781 Accuracy=78.32 LR=0.01
Test set: Average loss: 0.5182, Accuracy: 8244/10000 (82.44%)

EPOCH:46 Loss=0.40407708287239075 Batch_id=781 Accuracy=78.29 LR=0.01
Test set: Average loss: 0.5296, Accuracy: 8244/10000 (82.44%)

EPOCH:47 Loss=1.0935518741607666 Batch_id=781 Accuracy=78.60 LR=0.01
Test set: Average loss: 0.5289, Accuracy: 8242/10000 (82.42%)

EPOCH:48 Loss=0.3549344539642334 Batch_id=781 Accuracy=78.39 LR=0.01
Test set: Average loss: 0.4978, Accuracy: 8311/10000 (83.11%)

EPOCH:49 Loss=1.0633963346481323 Batch_id=781 Accuracy=78.57 LR=0.01
Test set: Average loss: 0.5033, Accuracy: 8307/10000 (83.07%)

EPOCH:50 Loss=0.31462806463241577 Batch_id=781 Accuracy=78.91 LR=0.01
Test set: Average loss: 0.4964, Accuracy: 8311/10000 (83.11%)

EPOCH:51 Loss=0.6347816586494446 Batch_id=781 Accuracy=79.09 LR=0.01
Test set: Average loss: 0.4925, Accuracy: 8353/10000 (83.53%)

EPOCH:52 Loss=0.8599880337715149 Batch_id=781 Accuracy=78.85 LR=0.01
Test set: Average loss: 0.5051, Accuracy: 8326/10000 (83.26%)

EPOCH:53 Loss=0.9188047647476196 Batch_id=781 Accuracy=79.28 LR=0.01
Test set: Average loss: 0.5337, Accuracy: 8251/10000 (82.51%)

EPOCH:54 Loss=0.4138769209384918 Batch_id=781 Accuracy=79.13 LR=0.01
Test set: Average loss: 0.4882, Accuracy: 8342/10000 (83.42%)

EPOCH:55 Loss=0.7534469366073608 Batch_id=781 Accuracy=79.34 LR=0.01
Test set: Average loss: 0.5078, Accuracy: 8302/10000 (83.02%)

EPOCH:56 Loss=0.627621591091156 Batch_id=781 Accuracy=79.40 LR=0.01
Test set: Average loss: 0.4733, Accuracy: 8372/10000 (83.72%)

EPOCH:57 Loss=0.8562610745429993 Batch_id=781 Accuracy=79.88 LR=0.01
Test set: Average loss: 0.5306, Accuracy: 8235/10000 (82.35%)

EPOCH:58 Loss=0.5065948367118835 Batch_id=781 Accuracy=79.50 LR=0.01
Test set: Average loss: 0.4934, Accuracy: 8342/10000 (83.42%)

EPOCH:59 Loss=0.4503714442253113 Batch_id=781 Accuracy=79.74 LR=0.01
Test set: Average loss: 0.4988, Accuracy: 8301/10000 (83.01%)

EPOCH:60 Loss=1.0231190919876099 Batch_id=781 Accuracy=80.18 LR=0.01
Test set: Average loss: 0.5053, Accuracy: 8286/10000 (82.86%)

EPOCH:61 Loss=0.8750219345092773 Batch_id=781 Accuracy=79.80 LR=0.01
Test set: Average loss: 0.4903, Accuracy: 8351/10000 (83.51%)

EPOCH:62 Loss=0.6153275370597839 Batch_id=781 Accuracy=79.84 LR=0.01
Test set: Average loss: 0.5026, Accuracy: 8339/10000 (83.39%)

EPOCH:63 Loss=0.6826620101928711 Batch_id=781 Accuracy=79.94 LR=0.01
Test set: Average loss: 0.4915, Accuracy: 8349/10000 (83.49%)

EPOCH:64 Loss=1.1233046054840088 Batch_id=781 Accuracy=80.43 LR=0.01
Test set: Average loss: 0.5269, Accuracy: 8267/10000 (82.67%)

EPOCH:65 Loss=0.6098084449768066 Batch_id=781 Accuracy=80.39 LR=0.01
Test set: Average loss: 0.4865, Accuracy: 8366/10000 (83.66%)

EPOCH:66 Loss=1.1618984937667847 Batch_id=781 Accuracy=80.65 LR=0.01
Test set: Average loss: 0.4877, Accuracy: 8349/10000 (83.49%)

EPOCH:67 Loss=0.40867191553115845 Batch_id=781 Accuracy=80.44 LR=0.01
Test set: Average loss: 0.4713, Accuracy: 8436/10000 (84.36%)

EPOCH:68 Loss=1.0472447872161865 Batch_id=781 Accuracy=80.45 LR=0.01
Test set: Average loss: 0.4810, Accuracy: 8403/10000 (84.03%)

EPOCH:69 Loss=0.6017810702323914 Batch_id=781 Accuracy=80.77 LR=0.01
Test set: Average loss: 0.4709, Accuracy: 8413/10000 (84.13%)

EPOCH:70 Loss=0.18660783767700195 Batch_id=781 Accuracy=80.53 LR=0.01
Test set: Average loss: 0.4786, Accuracy: 8383/10000 (83.83%)

EPOCH:71 Loss=0.6918805837631226 Batch_id=781 Accuracy=80.58 LR=0.01
Test set: Average loss: 0.4917, Accuracy: 8366/10000 (83.66%)

EPOCH:72 Loss=0.8374648094177246 Batch_id=781 Accuracy=80.65 LR=0.01
Test set: Average loss: 0.4877, Accuracy: 8395/10000 (83.95%)

EPOCH:73 Loss=0.7609348893165588 Batch_id=781 Accuracy=80.92 LR=0.01
Test set: Average loss: 0.4617, Accuracy: 8470/10000 (84.70%)

EPOCH:74 Loss=0.9469519853591919 Batch_id=781 Accuracy=81.08 LR=0.01
Test set: Average loss: 0.4755, Accuracy: 8411/10000 (84.11%)

EPOCH:75 Loss=0.8339831233024597 Batch_id=781 Accuracy=81.00 LR=0.01
Test set: Average loss: 0.4630, Accuracy: 8459/10000 (84.59%)

EPOCH:76 Loss=0.7702900767326355 Batch_id=781 Accuracy=81.02 LR=0.01
Test set: Average loss: 0.4711, Accuracy: 8439/10000 (84.39%)

EPOCH:77 Loss=0.562347412109375 Batch_id=781 Accuracy=80.83 LR=0.01
Test set: Average loss: 0.4877, Accuracy: 8396/10000 (83.96%)

EPOCH:78 Loss=0.7867209911346436 Batch_id=781 Accuracy=81.43 LR=0.01
Test set: Average loss: 0.4653, Accuracy: 8448/10000 (84.48%)

EPOCH:79 Loss=0.16648325324058533 Batch_id=781 Accuracy=81.29 LR=0.01
Test set: Average loss: 0.4628, Accuracy: 8492/10000 (84.92%)

EPOCH:80 Loss=0.3573480546474457 Batch_id=781 Accuracy=81.37 LR=0.01
Test set: Average loss: 0.4795, Accuracy: 8442/10000 (84.42%)

EPOCH:81 Loss=0.6302742958068848 Batch_id=781 Accuracy=81.51 LR=0.01
Test set: Average loss: 0.4803, Accuracy: 8401/10000 (84.01%)

EPOCH:82 Loss=0.4085537791252136 Batch_id=781 Accuracy=81.49 LR=0.01
Test set: Average loss: 0.4608, Accuracy: 8498/10000 (84.98%)

EPOCH:83 Loss=0.926561176776886 Batch_id=781 Accuracy=81.34 LR=0.01
Test set: Average loss: 0.4547, Accuracy: 8482/10000 (84.82%)

EPOCH:84 Loss=0.6663563251495361 Batch_id=781 Accuracy=81.30 LR=0.01
Test set: Average loss: 0.4620, Accuracy: 8467/10000 (84.67%)

EPOCH:85 Loss=0.4297630488872528 Batch_id=781 Accuracy=81.63 LR=0.01
Test set: Average loss: 0.4604, Accuracy: 8471/10000 (84.71%)

EPOCH:86 Loss=0.20247045159339905 Batch_id=781 Accuracy=81.75 LR=0.01
Test set: Average loss: 0.4798, Accuracy: 8435/10000 (84.35%)

EPOCH:87 Loss=0.45674511790275574 Batch_id=781 Accuracy=81.73 LR=0.01
Test set: Average loss: 0.4424, Accuracy: 8518/10000 (85.18%)

EPOCH:88 Loss=0.6810380220413208 Batch_id=781 Accuracy=81.86 LR=0.01
Test set: Average loss: 0.4711, Accuracy: 8453/10000 (84.53%)

EPOCH:89 Loss=0.9011926651000977 Batch_id=781 Accuracy=81.79 LR=0.01
Test set: Average loss: 0.4553, Accuracy: 8499/10000 (84.99%)

EPOCH:90 Loss=0.37621384859085083 Batch_id=781 Accuracy=81.90 LR=0.01
Test set: Average loss: 0.4686, Accuracy: 8472/10000 (84.72%)

EPOCH:91 Loss=0.710060179233551 Batch_id=781 Accuracy=82.01 LR=0.01
Test set: Average loss: 0.4681, Accuracy: 8447/10000 (84.47%)

EPOCH:92 Loss=0.5635992288589478 Batch_id=781 Accuracy=82.03 LR=0.01
Test set: Average loss: 0.4462, Accuracy: 8507/10000 (85.07%)

EPOCH:93 Loss=1.0358370542526245 Batch_id=781 Accuracy=81.88 LR=0.01
Test set: Average loss: 0.4493, Accuracy: 8498/10000 (84.98%)

EPOCH:94 Loss=0.5407629609107971 Batch_id=781 Accuracy=82.17 LR=0.01
Test set: Average loss: 0.4454, Accuracy: 8508/10000 (85.08%)

EPOCH:95 Loss=0.4008288085460663 Batch_id=781 Accuracy=82.02 LR=0.01
Test set: Average loss: 0.4651, Accuracy: 8491/10000 (84.91%)

EPOCH:96 Loss=0.6344500184059143 Batch_id=781 Accuracy=81.93 LR=0.01
Test set: Average loss: 0.4497, Accuracy: 8491/10000 (84.91%)

EPOCH:97 Loss=1.00767183303833 Batch_id=781 Accuracy=82.34 LR=0.01
Test set: Average loss: 0.4492, Accuracy: 8497/10000 (84.97%)

EPOCH:98 Loss=0.3183903098106384 Batch_id=781 Accuracy=82.20 LR=0.01
Test set: Average loss: 0.4423, Accuracy: 8546/10000 (85.46%)

EPOCH:99 Loss=0.3362240791320801 Batch_id=781 Accuracy=82.04 LR=0.01
Test set: Average loss: 0.4469, Accuracy: 8513/10000 (85.13%)

Model saved in .pt format to CIFAR_10_model.pt

```
