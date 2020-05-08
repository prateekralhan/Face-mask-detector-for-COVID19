# Face-mask-detector-for-COVID19
Implemented a simple yet powerful Face mask detector for COVID 19.

## Dependencies:
1. numpy
2. argparse
3. openCV
4. keras/tensorflow
5. matplotlib
6. sklearn
7. imutils
Simply run the command: ***pip install <package_name>*** to install the dependencies to your system.

Note: 
For openCV installation, refer [this.](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)

## Dataset:
A big shoutout to [Prajna Bhandary](https://github.com/prajnasb/observations) for getting the face mask dataset. I have ensembled the dataset [here](https://drive.google.com/drive/folders/1s7uT7YIs-wCmJaf8a-Pxn1A99NaMJw6B?usp=sharing)
This dataset consists of 1,784 images belonging to two classes:
* with_mask
  i. 1098 images
* without_mask
  i. 686 images
(I need to add more images in to **without_mask** sub folder to prevent the slight imbalance here.)
The dataset has been created using normal images of faces and then creating a custom computer vision Python script to add face masks to them, thereby creating an artificial (but still real-world applicable) dataset.

## Phases of the face mask detector:
1. Phase I
  a. Load face mask dataset.
  b. Train face mask classifier with Keras/Tensorflow.
  c. Serialize face mask classifier to disk.
2. Phase II
  a. Load face mask classifier from disk.
  b. Detect face(s) in image/video stream.
  c. Extract each face ROI
  d. Apply face mask classifier to each face ROI to determine **mask** or **no mask**.
  e. Show results
  
 ## Usage:
 1. Download and extract the **dataset** folder in the same directory of this project.
 2. Execute the command: **python train_mask_detector.py --dataset dataset** to start training your classifier. I have fine tuned the [MobileNet V2 Architecture](https://arxiv.org/abs/1801.04381) here. The classifier can be further refined using other SOTA models like ResNet101, ResNet152, InceptionV3 models etc.
 3. After the training ends, an image showcasing the training process will be saved in your directory :
 
