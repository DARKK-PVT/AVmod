# AVmod
AVmod is a Audiovisual modulator developed as Minor Project for junior year

* AVmod uses LSGAN and cyclic GAN to achieve the functionality of face swaping and voice modulation

## Description
* MTCNN_video_face_detection_aligmnet.ipynb
  * File-1 
  * Responsible for frame extraction, face detection/alignment on input video.
  * Detected faces are saved in ./faces/raw_face for non-aligned result and ./faces/aligned_faces for aligned results.
  * Crude eyes binary masks saved in ./faces/binary_mask_eye.
  
* prep_binary_masks.ipynb
  * File-2 
  * For datapreprocessing.
  * Create binary masks using aligned_faces and save results in ./binary_masks/faceA_eyes and ./binary_masks/faceB_eyes folder.
  * Require face_alignment package.
  
* train.ipynb
  * File-3 
  * Used for model training.
  * Require additional training images generated through prep_binary_masks.ipynb
  * Save models in ./model directory.
  * Save backup models in ./model/backup_iter{iteration_num}.

* video_conversion.ipynb
  * File-4 
  * Used for video conversion based on training done in train.ipynb
  * Use  five-points landmarks for face alignment.

## Training Data
* Pick images that are stored in ./facesA/aligned_faces and ./facesB/aligned_faces for each target.
* Resizing of image will be performed to make images 256x256 for training.
* Training will happen for 40000 iterations (default) can be increased to 80000 and more according to requirement.


## Requirements
* python 3.6.4
* tensorflow r1.15.2
* keras r2.1.5
* opencv
* keras_vggface
* moviepy
* face_alignment
* pathlib

## Todo
* Functionality for voice modulation
* Increase face swapping area
* Binary Mask for mouth
* Interface for easy access to training and conversion
* GPU support for py files (CUDA)

## Acknowledgment
Code borrowed from tjwei, eriklindernoren, fchollet and keras-contrib. The generative network is adopted from CycleGAN. Weights and scripts of MTCNN are from FaceNet
