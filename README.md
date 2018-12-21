# Image-Segmentation-using-open-CV-and-KL-divergence
Python GUI using open CV to select tree logs for segmentation

#Objective: Enable user to manually segment the tree logs in an image using a GUI with priors which can be controlled by mouse keys and movements.
#Tools and techniques: Open CV, python 3.x, KL diivergence for segmentation
#Functionality:
The program opens the images from 'Train Images' folder in a GUI window and moves it to the appropriate folder after segmentation process.
The GUI shows a square prior around mouse pointer which can be increased or decreased in size using mouse wheel.
The size of the prior is customized for our problem statement (tree logs on a truck). The size can be easily modified. 
#Mouse Usage:
'Left click' to select a log to segment and apply marker.
'Mouse wheel' to control the size of the square prior
'Right double click' to see the segmented masks at any point
#Keyboard Usage:
'D' or 'd' to delete the marker on the selected log
'S' or 's' to save the segmented mask in the destined folder
'N' or 'n' to open the next image
'R' or 'r' to reset the image to the original image.
#######################################################################################################################################
Important Note: User needs to create 4 folders and change the hardcoded folders in the program(main loop) before running it.
1.  input_path  = '.../Train images/'                       -> Input raw images. Will be moved to output folder if segmented or to unused folder if ignored.
2.  output_path = '.../Train images/processed/train/image/' -> The folder where the (raw) training images are stored after manual segmentation process
3.  label_path  = '.../Train images/processed/train/label/' -> The folder where the training masks are stored after segmentation process
4.  unused_path = '.../Train images/processed/unused/'      -> The ignored, unclean images are moved to this folder.
#######################################################################################################################################
Disclaimer: The sample images used in the repository are from google and may be subject to copyright. The images are purely used for demo.

We used the tool to segment tree logs in 390 images, trained a basic u-net with image augmentation and predicted the logs for a test set of 30 images with an IOU of 80%. 
