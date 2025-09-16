# DBU-Net
Dual-Branch U-Net for medical image segmentation


Note that the paths for input_dir and output_dir are both `../`

The resulted file structure is as follows.

```
├── DBU-Net
├── inputs
│   ├── busi
│     ├── images
│           ├── malignant (1).png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── malignant (1)_mask.png
|           ├── ...
│   ├── GLAS
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
│   ├── CVC-ClinicDB
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
```



You can simply train DBU-Net on a single GPU by specifing the dataset name --dataset

```
cd DBU-Net
python train.py --dataset <datasetName> --name <any>
```

For example

```
python train.py --dataset busi --name busi_DBU-Net
```

And You can use the following command to evaluate the trained model.

```
python val.py --name busi_DBU-Net
```



