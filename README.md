# HGNN-FSL

Code for paper *Hierarchical Graph Neural Networks for Few-Shot Learning*.[pdf]()

## Abstract



## Citation

If you find this code useful you can cite us using the following bibTex:

```

```

## Requirements

* python 3.6.9
* pytorch 1.2.0
* torchvision  0.4.0
* tensorboardx
* numpy
* pandas
* tqdm

## Dataset

### Mini-Imagenet

You can download miniImagenet dataset from EGNN's author [here](https://drive.google.com/drive/folders/15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w)

Copy them inside following directory:

 ```
.
├── ...
└── dataset
	└── compacted_datasets
		├── mini_imagenet_train.pickle
		├──	mini_imagenet_val.pickle
		└── mini_imagenet_test.pickle 
 ```

### Tiered-Imagenet

You can download tieredimagenet dataset from few-shot-ssl-public's author [here](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view)

Copy them inside following directory:

```
.
├── ...
└── dataset
	└── tiered-imagenet
		├── train_images_png.pkl
		├── train_labels.pkl
		├── val_images_png.pkl
		├── val_labels.pkl
		├── test_images_png.pkl
		├── test_labels.pkl
		├── class_names.txt
		└── synsets.txt
```

## Training

```
# ************************** miniImagenet, 5way 5shot *****************************
$ python train.py --device cuda:0 --dataset mini --num_ways 5 --num_shots 5 --transductive True --pool_mode kn --unet_mode addold
$ python train.py --device cuda:0 --dataset mini --num_ways 5 --num_shots 5 --transductive False --pool_mode kn --unet_mode addold

# ************************** miniImagenet, 5way 1shot *****************************
$ python train.py --device cuda:0 --dataset mini --num_ways 5 --num_shots 1 --transductive True --pool_mode kn --unet_mode addold
$ python train.py --device cuda:0 --dataset mini --num_ways 5 --num_shots 1 --transductive False --pool_mode kn --unet_mode addold

# ************************** tieredImagenet, 5way 5shot *****************************
$ python train.py --device cuda:0 --dataset tiered --num_ways 5 --num_shots 5 --transductive True --pool_mode kn --unet_mode addold
$ python train.py --device cuda:0 --dataset tiered --num_ways 5 --num_shots 5 --transductive False --pool_mode kn --unet_mode addold
```

## Evaluation

The trained models are saved in the path './asset/checkpoints/', with the name of 'D-{dataset}_ N-{ways} _K-{shots} _Q-{num_queries} _B-{batch size} _T-{transductive} _P-{pooling mode} _Un-{unet mode}'. So, for example, if you want to test the trained model of 'miniImagenet, 5way 5shot, transductive, kngpooling, addold' setting, you can give --test_model argument as follow:

```
$ python3 eval.py --test_model D-mini_N-5_K-5_Q-5_B-40_T-True_P-kn_Un-addold
```

## Result

You can download our experiment results and trained models from [here](https://drive.google.com/drive/u/0/folders/1pRbit4P_MAjwL4BdSNwsGHthxinLCzF-)

### **miniImagenet,non-tranductive**

|   Model    | 5-way 5-shot acc(%) |
| :--------: | :-----------------: |
|    GNN     |        66.41        |
|    EGNN    |        66.85        |
| (ours)HGNN |      **69.05**      |

### miniImageNet, transductive

|   Model    | 5-way 5-shot acc(%) |
| :--------: | :-----------------: |
|    GNN*    |        75.41        |
|    EGNN    |        76.37        |
| (ours)HGNN |      **79.64**      |



### tieredImageNet, non-transductive

|   Model    | 5-way 5-shot acc(%) |
| :--------: | :-----------------: |
|    GNN     |                     |
|    EGNN    |        70.98        |
| (ours)HGNN |      **73.01**      |

### tieredImageNet, transductive

|   Model    | 5-way 5-shot acc(%) |
| :--------: | :-----------------: |
|    GNN*    |                     |
|    EGNN    |        80.15        |
| (ours)HGNN |      **83.34**      |

GNN transductive mode was implemented in [here](https://github.com/gaieepo/few-shot-gnn) by gaieepo.