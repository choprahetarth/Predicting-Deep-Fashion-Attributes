# Predicting-Deep-Fashion-Attributes

Hello! 
This notebook uses the dataset provided by FlixStock, as per their deep learning assessement. The dataset consisted of a csv file having the images path and their attributes. It was accompanied by an images folder consisting of all the images.
The following approach was followed in order to train and predict on the given images


1.   The "attributes.csv" was read as a dataframe in Pandas
2.   There were some duplicated image names in the dataframe, which did not exist in the images folder. The first row was kept, and the subsequent ones were removed.
3.   The csv file was inspected for "N/A" files and decision was taken by me to impute the "N/A" files as "-1", treating it as a category.
4.   The distributuion of the multiple lables in different classes were checked. The problem was identified to be a rather skewed multi-class multi-label classification.
5.   One hot encoding was done for all the classes and all the labels by using "get_dummies" of Pandas.
6.   The Folder was inspected of a "Thumb.db" file, it was manually removed.
7.   The data was split into training/testing and validation datasets by creating a random mask in Pandas.
8.   The dataset was created by tweaking an example in PyTorch's official documentation ([Dataset Creation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)). 
  *   Multiple Transforms were used for all three types of datasets, including random flip, vertical flip, and the images were resized to 512,512 (even though experiments were performed with 256,256 also).
  *   The labels and images were returned as tensors, outputs and their shapes were checked with all the three dataloaders. The batch size was taken as 16 for the training and validation dataset. No batch size was taken for the test. 
9.    Transfer Learning was the approach chosen. Two models were primarily experimented with (after freezing the models and using them just as feature extractors) - 
  *   Resnet18 (Following this tutorial -> [Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#further-learning))
  *   Resnext50_32x4d - Which tends to perform slightly better.
10.   The training loop was inspired from the above tutorial, and was tweaked according to our needs. The following changes were made- 
  *   Since the problem was a multiclass, multilabel one, the outputs of the frozen layer were sent to a sigmoid function, and after that BCE loss was used to score the loss. Inspiration for the tweak arose from the following link ->[Stats Exchange](https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n)
  *   A fully connected layer was provided with 21 (total dummies) outputs, which provided the predicted values. 
  *   Three metrics were used primarily-> loss, accuracy and f1 score.
  *   Adam and SGD were both experimented with, leading to the selection of Adam because it gave better performance.    
11.   The Model was finally visualized on the Test Dataset, and on visual inspection it showed decent results.
