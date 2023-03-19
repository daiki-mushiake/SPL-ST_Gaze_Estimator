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

## Code & Procedure
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
Automatically performs an evaluation every 500steps.

# 5) Check style transferred images
Please, check style transferred images in '/eval_img' and 'refine_img'.
'''
## Code & Procedure
'''
Pre-trained weights is **"//".**
     
```

