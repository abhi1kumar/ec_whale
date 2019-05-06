This project uses Siamese networks and Classification based approach to identify a whale image. The dataset used for this project is available from [Kaggle](https://www.kaggle.com/c/whale-categorization-playground/) 

## Requirements
1. Python3
2. [Pytorch](http://pytorch.org/)
3. Torchvision


## Usage
- Data is to be downloaded in a directory named data which you have to make in the git folder. An example of sample relative image path 
 data/train/d9a83d92.jpg

- Install all the libraries.

- Type in terminal
```bash
python3 train_test_whale_siamese_run_15.py
```
If you have multiple GPUs and you want to use a single GPU, type
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_test_whale_siamese_run_15.py
```

## Interpretation
Output is shown as a ratio. Multiply by 100 to interpret it as a percentage. eg: if val acc is shown 0.1234 it means the accuracy is 12.34%

## Authors
1. Abhinav Kumar, University of Utah
2. Surojit Saha, University of Utah

## Acknowledgements
[Prof Tolga Tasdizen](http://www.sci.utah.edu/~tolga/ResearchWebPages/index.html)

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
