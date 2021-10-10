# Deep-Learning-Assignment-3-Parsing-data-and-Regularization
This is my final submission for Homework 3 in Deep Learning for Advanced Robot Perception. Please find the code repository at: https://github.com/nastallings/Deep-Learning-Assignment-3-Parsing-data-and-Regularization

This assignment had two parts:


Part 1: Parsing Data
This part of the assignment is completed in parsing_data.py. The code parses the VOC PASCAL 2012 dataset and creates images for each extracted element and stores it in the appropriate folder. The intermediate data set produce is called Intermediate_Data_Set and a binary version of this directory is created called Intermediate_Data_Set.zip

Due to size, the VOC PASCAL data set, the Intermediate_Data_Set, and the Intermediate_Data_Set.zip are not part of the git repository. 

To get the VOC PASCAL dataset, download it from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
The Intermediate_Data_Set and the Intermediate_Data_Set.zip can be obtained by running the code on the downloaded dataset. 
Make sure to extract the VOC PASCAL dataset to a folder called Data in the project directory. See VOC_PASCAL_Hierarchy.png for an example.



Part 2: Building a Deep Learning model with dropout
This part of the assignment is done in three files: baseline.py, dropout_hidden.py, and dropout_visbile.py. These models can be run and the best-obtained performance is displayed. A report discussing the steps taken to obtain these results showing the accuracy and loss of the experiment can be found in: Stallings_HW3_Report.pdf


