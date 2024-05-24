# Semenatic Segmentation using U-Net

by Utsav Negi

## Aim
The aim of this report is to demonstrate the approach used to implement multi-instance semantic
segmentation on Purdue Shapes Dataset Objects and single-instance semantic segmentation on COCO
Dataset Objects by training model based on Unet Architecture.

## Source Code Detail
For this assignment, the details about the source code are as follows:
<ul>
  <li>ss_coco.py: Source code consisting of all the essential classes, functions & main ML pipeline
for single instance semantic segmentation on COCO Dataset.</li>
  <li>ss_purdueshapes.py: Source code consisting of all the essential classes, functions & main ML pipeline
for multi-instance semantic segmentation on PurdueShapes Dataset.</li>
</ul>

## Unet Model Architecture

The Unet Model is a commonly used model for semantic segmentation. The model has two phases:
encoder phase and decoder phase. In the encoder phase, the size of the input image gets smaller which
results in high levels of feature abstraction. In the decode phase, the determined feature abstractions are
mapped to each pixel in the image. The code uses SkipBlockDN to encode the image and SkipBlockUP to
decode the image. Furthermore, both phases are inter-connected using SkipConnections at each stage
of respective encoding and decoding. The result produced by the model is a mask which segregates the
object pixels from the background pixels. This model is being used to implement multi-instance
semantic segmentation on Purdue Shapes Dataset Objects and single-instance semantic segmentation
on COCO Dataset Objects.

## Semantic Segmentation on Purdue Shapes Dataset Objects

As provided in the assignment requirements, the model for semantic segmentation on the Purdue Shape
Dataset Objects is trained on three different loss criteria: Mean Square Error Loss, Dice Loss, and a
combination of scaled Dice Loss and Mean Square Error Loss. The model parameters are optimized
using SGD optimizer with a learning rate of 1e-4 and 0.9 as momentum hyperparameter. The training
epoch is set to 6 with a batch size of 6. At the end of training with different loss functions, the training
function returns a list of average loss incurred during training.

## Semantic Segmentation on COCO Dataset Objects

The model uses COCO Train2017 dataset for training the model and COCO Val2017 dataset for
evaluating the performance of the model. The COCO Train2017 and COCO Val2017 images and their
respective annotations are downloaded to the working directory. Based on the given categories: dog,
cake, and motorcycle, two dictionaries, dedicated for storing training data and validation data, are
created which map each category to an array of image data belonging to that category. Furthermore, each
image data stored in a triplet consisting of image tensor which is resized to 256 x 256, labels of the
objects present in the image, and the scaled bounding box tensors of the objects present in the image.
The function getData() is responsible for executing the above tasks and it returns the training data
dictionary and the validation data dictionary.

The semantic segmentation is conducted on Motorcycle, Dog and Cake categories of COCO
Dataset Objects. At the start of the pipeline, the image data along with its binary mask are extracted using
the COCO API which are resized to 256 x 256 and are stored in two dictionaries: one for training data and
another for validation data. These dictionaries are used to create the ImageDataset class which is a child
class of torch.utils.data.Dataset class. For this purpose, the same Unet model is used with some minor
tuning at the output layer to match the output dimensions with dimensions of the ground truth binary
mask. The model is trained using MSE Loss, Dice Loss and a combination of MSE Loss and scaled Dice
Loss to compare its performance on COCO Dataset Objects. The model parameters are optimized using
SGD optimizer with a learning rate of 5e-4 and 0.95 as momentum hyperparameter. The training epoch is
set to 15 with a batch size of 25. At the end of training with different loss functions, the training function
returns a list of average loss incurred during training.
