# Draw_Guess

## ToDO list

Requirement file is provided [here](https://docs.google.com/document/d/1olgVdN7WWXoFBzTxik3T2vjcWCKDIGKf1-pUFZ3PUxI/edit)

- [x] 2/18 Forming teams
- [x] 3/9 - 3/16 Project Check-in #1 
- [x] 3/18 Project Proposal
- [ ] 4/8 - 4/15 Project Check-in #2 
  - [x] main pipeline for classification
    - [x] Dataloader
    - [x] classification metrics
    - [x] vanilla classifier (ResNet)
  - [ ] field study of sketch recognition
  - [ ] brainstorm of methodology
  - [ ] brainstorm of metrics
  - [ ] brainstorm of ethics
  - [ ] write up outlines
- [ ] 4/20 - 4/27 Project Check-in #3
- [ ] 5/2 - 5/9 Final Check in
- [ ] 5/11 Final Project Due
- [ ] 5/13 Deep Learning Day 2022


## Env Setting
**System**: Ubuntu 18.04 (Linux)

**GPU**: NVIDIA GeForce 3090 * 1

**PyTorch CUDA Toolkit**: 11.3
 
### Installation
```something here```


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

### Sample: 1 image - 5 sketchs
<table>
  <tr>
    <td> <img src="demo/n02691156_58.jpg"  alt="1" width = 256px height = 256px align="center"></td>
    <td> <img src="demo/n02691156_58-1.png"  alt="2" width = 256px height = 256px ></td>
   </tr> 
   <tr>
    <td><img src="demo/n02691156_58-2.png" alt="3" width = 256px height = 256px></td>
      <td><img src="demo/n02691156_58-3.png" alt="4" width = 256px height = 256px></td>
    </tr>
    <tr>
      <td><img src="demo/n02691156_58-4.png" alt="5" width = 256px height = 256px>
      <td><img src="demo/n02691156_58-5.png" align="right" alt="6" width = 256px height = 256px>
  
  </tr>
</table>

### Augmentations within 'photo'
<table border="1">
    <tr>
        <td> aug_id </td>
        <td> description </td>
    </tr>
    <tr>
        <td> tx_000000000000 </td>
        <td> image is non-uniformly scaled to 256x256 </td>
    </tr>
    <tr>
        <td> tx_000100000000 </td> 
        <td> image bounding box scaled to 256x256 with
                      an additional +10% on each edge; note 
                      that due to position within the image,
                      sometimes the object is not centered </td>
    </tr>
</table>

### Augmentations within 'sketch'
<table border="1">
    <tr>
        <td> aug_id </td>
        <td> description </td>
    </tr>
    <tr>
        <td> tx_000000000000 </td>
        <td> sketch canvas is rendered to 256x256
                      such that it undergoes the same
                      scaling as the paired photo </td>
    </tr>
    <tr>
        <td> tx_000000000010 </td> 
        <td> sketch is translated such that it is 
                      centered on the object bounding box </td>
    </tr>
    <tr>
        <td> tx_000000000110 </td> 
        <td> sketch is centered on bounding box and
                      is uniformly scaled such that one dimension
                      (x or y; whichever requires the least amount
                      of scaling) fits within the bounding box </td>
    </tr>
    <tr>
        <td> tx_000000001010 </td> 
        <td> sketch is centered on bounding box and
                      is uniformly scaled such that one dimension
                      (x or y; whichever requires the most amount
                      of scaling) fits within the bounding box </td>
    </tr>
    <tr>
        <td> tx_000000001110 </td> 
        <td> sketch is centered on bounding box and
                      is non-uniformly scaled such that it 
                      completely fits within the bounding box </td>
    </tr>
    <tr>
        <td> tx_000100000000 </td> 
        <td> sketch is centered and uniformly scaled 
                      such that its greatest dimension (x or y) 
                      fills 78% of the canvas (roughly the same
                      as in Eitz 2012 sketch data set) </td>
    </tr>

</table>


## Reference

ToDO