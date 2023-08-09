=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─ThreeConvBlock: 1-1                    --
|    └─Sequential: 2-1                   --
|    |    └─Conv2d: 3-1                  1,728
|    |    └─BatchNorm2d: 3-2             128
|    |    └─ReLU: 3-3                    --
|    |    └─Conv2d: 3-4                  36,864
|    |    └─BatchNorm2d: 3-5             128
|    |    └─ReLU: 3-6                    --
|    |    └─Conv2d: 3-7                  36,864
|    |    └─BatchNorm2d: 3-8             128
|    |    └─ReLU: 3-9                    --
├─DownStage: 1-2                         --
|    └─Sequential: 2-2                   --
|    |    └─MaxPool2d: 3-10              --
|    |    └─ThreeConvBlock: 3-11         369,408
├─DownStage: 1-3                         --
|    └─Sequential: 2-3                   --
|    |    └─MaxPool2d: 3-12              --
|    |    └─ThreeConvBlock: 3-13         1,476,096
├─DownStage: 1-4                         --
|    └─Sequential: 2-4                   --
|    |    └─MaxPool2d: 3-14              --
|    |    └─ThreeConvBlock: 3-15         5,901,312
├─DownStage: 1-5                         --
|    └─Sequential: 2-5                   --
|    |    └─MaxPool2d: 3-16              --
|    |    └─ThreeConvBlock: 3-17         23,599,104
├─UpStage: 1-6                           --
|    └─Upsample: 2-6                     --
|    └─ThreeConvBlock: 2-7               --
|    |    └─Sequential: 3-18             9,440,256
├─UpStage: 1-7                           --
|    └─Upsample: 2-8                     --
|    └─ThreeConvBlock: 2-9               --
|    |    └─Sequential: 3-19             2,360,832
├─UpStage: 1-8                           --
|    └─Upsample: 2-10                    --
|    └─ThreeConvBlock: 2-11              --
|    |    └─Sequential: 3-20             590,592
├─UpStage: 1-9                           --
|    └─Upsample: 2-12                    --
|    └─ThreeConvBlock: 2-13              --
|    |    └─Sequential: 3-21             147,840
├─Conv2d: 1-10                           65
=================================================================
Total params: 43,961,345
Trainable params: 43,961,345
Non-trainable params: 0
=================================================================