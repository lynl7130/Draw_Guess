# Draw_Guess

## Env Setting
**System**: Ubuntu 18.04 (Linux)

**GPU**: NVIDIA GeForce 3090 * 1

**PyTorch CUDA Toolkit**: 11.3
 



## Data

Download Sketchy Database from [here](https://sketchy.eye.gatech.edu/)


### Raw Data Structure:

```bash
├── 256x256
│   ├── photo
│   │   ├── tx_000000000000
|   |   |    ├── airplane
|   |   |    |    ├── n02691156_58.jpg
|   |   |    |    ├── n02691156_196.jpg
|   |   |    |    ├── ...
|   |   |    ├── alarm_clock
|   |   |    ├── ...
|   |   └── tx_000100000000
|   |   |    ├── ...
│   └── sketch
│       ├── tx_000000000000
|       |    ├── airplane
|       |    |    ├── n02691156_58-1.png
|       |    |    ├── n02691156_58-2.png
|       |    |    ├── n02691156_58-3.png
|       |    |    ├── n02691156_58-4.png
|       |    |    ├── n02691156_58-5.png
|       |    |    ├── n02691156_196-1.png
|       |    |    ├── ...
|       |    ├── ...
│       ├── tx_000000000010
|       |    ├── ...
│       ├── tx_000000000110
|       |    ├── ...
│       ├── tx_000000001010
|       |    ├── ...
│       ├── tx_000000001110
|       |    ├── ...
│       └── tx_000100000000
|            ├── ...
└── README.txt
```