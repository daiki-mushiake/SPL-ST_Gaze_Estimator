# Shape Preserving by Dense Pixel Correspondences in Image Style Transfer for Gaze Estimation

![example figure](image/method.jpg)
<!-- **Ours methhod.** -->


## Directories
```plain
SPL-ST_Gaze_Estimator/
	data/
	lib/
	Refiner1_checkpoint_path/
	augmentation_img/
	check_image/
	checkpoint_path
	eval_img/
	image/
	log/
	raft_pretrained_model/
	refine_img/
	result/
	test_output/
	valid_ldmk/
       	     				
```


## Setup
- Install required packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
- Download [MPIIGAZE dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) 
[UT Multi-view dataset](https://www.ut-vision.org/datasets/) 
[Columbia Gaze(CAVE) dataset](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) 
[ETH-XGaze dataset](https://ait.ethz.ch/projects/2020/ETH-XGaze/) 

## Code
1. Data preparation following MPIIGaze & UT Multiview-dataset. <br> The following procedure is actually required.

```bash

# 1) preprocess real image dataset
Preprocess dataset original images using the files in the 'preprocess dataset' or download the dataset(/misc/dl001/dataset/gaze_dataset).

# 2) preprocess synthetic images dataset
Generate synthetic images in Unityeyes and pre-process it using the file (/preprocess/unityeyes_preprocess.ipynb) or download the dataset.
UnityEyes(https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/)

# 3) Train Style Transfer & Gaze Estimator
CUDA_VISIBLE_DEVICES=, python lib/train_gaze_estimator.py

# 4) Evaluation
Automatically performs an evaluation every 500steps

# 5) Check style transferred images
Please, check style transferred images in '/eval_img' and 'refine_img'.

     
```

2. Gaze Estimation
```bash
CUDA_VISIBLE_DEVICES=8,9 python3 train_gaze_estimator.py  # train & evaluation
```
3. Generate mapped radar data (MRD) <br> 
The following procedure is actually required, but can be avoided by accessing  <br> 
**"/kotani/workspace/DERURD/data/mer_2_30_5_0.5_mlp2_aug4_neg_0.0.h5".**

```bash
python cal_mer.py
```

4. Train depth completion by using the enhanced depth
- Depth completion scheme 1 ([Using depths and RGB as input channels](https://arxiv.org/pdf/1709.07492.pdf))

```bash
python train_depth.py        	# train
python test_depth.py         	# test
```
<!-- Download [pre-trained weights](https://) -->
Pre-trained weights is **"/kotani/workspace/DERURD/data/train_data/prepared_data.h5".**

- Depth completion scheme 2 ([Multi-Scale Guided Cascade Hourglass Network](https://github.com/anglixjtu/msg_chn_wacv20))

```bash
python train_depth_hg.py        # train
python test_depth_hg.py         # test
```
<!-- Download [pre-trained weights](https://). -->
Pre-trained weights is **"/kotani/workspace/DERURD/data/train_data/prepared_data.h5".**


## Citation
```plain

```


# SPL-ST_Gaze_Estimator
