# Auto-Encoder

- Create your python virtual environment by

  ```bash
  conda create --name unet python=3.10 # create a virtual environment called "unet" with python version 3.10
  ```

- **Install Pytorch**: Please follow link [here]([Get Started](https://pytorch.org/get-started/locally/)).

​	(PyTorch version >= 1.5.1)

- **Other third-party packages**: You can use pip to install the dependencies by

```bash
pip install -r requirements.txt
```

## Train

- [LGG Brain Segmentation]([Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation))

  ```
  kaggle_3m 
  	|_TCGA_CS_4941_19960909
  	|_TCGA_CS_4942_19970222
  	...
  ```

  ### Usage

  - Training model

  ```bash
  python lgg_main.py
  ```

  - Check the results

  ```bash
  tensorboard --logdir runs
  ```

- [DRIVE]([Introduction - DRIVE - Grand Challenge](https://drive.grand-challenge.org/))

-  [BR35H(Brain Tumor Segmentation)](https://www.kaggle.com/datasets/sushreeswain/brain-tumor-segmentation)

# 
