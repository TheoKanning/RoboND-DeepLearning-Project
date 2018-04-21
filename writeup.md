## Project: Deep Learning

[//]: # (Image References)

[people]: ./misc_images/people.png
[walls]: ./misc_images/walls.png

### Fully Convolutional Networks
#### Theory
Fully Convolutional Networks (FCNs) are a type of neural network composed entirely of convolutional layers. Typical 
convolutional networks, like those used in image recognition, end with a series of fully-connected layers, which means
that their outputs don't maintain the spatial information of their inputs. In contrast, FCNs return an output that is 
the same size as their input, with each pixel being classified individually.

Because FCNs perform classification for each pixel, they are used for Semantic Segmentation, the process of separating
an image into classes. Unlike bounding boxes, semantic segmentation returns classification of arbitrary shape and size.

#### Architecture
FCNs consist of two main steps: Encoding and Decoding. Encoding is the process of applying convolutional layers and shrinking
the data from the input size into a smaller representation. Decoding is the opposite, using bilinear upsampling combined
with more convolution to restore the data to its original size.

#### 1x1 Convolutions
In order to maintain spatial data, FCNs replace fully-connected layers with 1x1 convolutions in order to maintain spatial 
relationships.

#### Separable Convolutions
Separable convolutions split a convolutional layer into two steps. The first applies a MxN kernel without changing the 
number of layers, and the second applies a 1x1 kernel that transforms the number of layers. This results in a dramatic
reduction in layer weights, which increases training speed, reduces model size, and prevents overfitting.

### Network Architecture
My semantic segmenter architecture consisted of two encoder blocks, a 1x1 convolution, and 2 decoder blocks.

| Block | Layer |
---|---
 Encoder 1 | Conv2d, stride = 1, kernels = 16 
 Encoder 1 | Conv2d, stride = 2, kernels = 16
 Encoder 2 | Conv2d, stride = 1, kernels = 32
 Encoder 2 | Conv2d, stride = 2, kernels = 32
 1x1 Convolution | Conv2d, stride = 1, kernels = 64
 Decoder 1 | Bilinear Upsample
 Decoder 1 | Skip Connection to Encoder 1
 Decoder 1 | Conv2d, stride = 1, kernels = 32
 Decoder 1 | Conv2d, stride = 1, kernels = 32
 Decoder 2 | Bilinear Upsample
 Decoder 2 | Skip Connection to Input
 Decoder 2 | Conv2d, stride = 1, kernels = 16
 Decoder 2 | Conv2d, stride = 1, kernels = 16
 Output | Conv2d, stride = 1, filters = 3
 


#### Encoder Block
Each encoded block consists of two convolutions, each separable with batch normalization. The first uses a stride of 1 to
maintain the layer size, and the second uses a stride of 2 to reduce each dimension by a factor of 2. I added the first
convolution after reading about the VGG16 architecture, which has many convolutions that don't change the output shape.

#### Decoder Block
Each decoder block consists of bilinear upsampling and two convolutions. Bilinear upsampling doubles each dimension of the
output shape by using the average of each cell's neighbors. Each convolution is done with a stride of 1 to avoid changing
the output shape.

#### Skip Connections
In order to combine high-res inputs with processed data, each decoder block also included a skip connection from the encoder
layer of matching size. This was done by concatenating the layers in Keras, and otherwise required no modification of the
network.

### Training
All training was done locally on my GPU, which worked well despite being almost five years old. Each of my training runs 
took no more than 30 minutes.

#### Data Collection
Before training, I gathered some more data in the simulator. Since many of the provided images contained few people, I 
created a scenario with dozens of people and captured about 400 images. 

![People][people]

I then noticed that my network had a tendency to mistake dark gray walls for bystanders, so I gathered a few hundred 
images while exploring uninhabited areas.
![walls][walls]

#### Hyperparameters
My final hyperparameters were as follows. The number of epochs was determined by checking the model's performance every 
5 epochs, and stopping when no more improvement was made. Batch size was made as large as possible without running out 
of memory.

- learning_rate = 0.01 A good default, this worked fine so I never changed it.
- num_samples = 4750
- batch_size = 30 The most that would run on my gpu without running out of memory
- num_epochs = 20 I noticed no improvement at 25, so I stopped here
- steps_per_epoch = int(num_samples/batch_size) Each image once per epoch
- validation_steps = 50

### Results
My model achieved and IOU of 0.84 while following the hero, 0.23 while far away, and a final score of 0.41.

### Future Enhancements
My model (predictably) did poorly when the hero was far away. I didn't specifically capture any data while far away from 
the hero, so adding more training data here would likely help.

FCNs are great candidates for transfer learning, so I might be able to improve performance by using a pre-trained encoder
from VGG16 or another network. Several well-known networks are built into Keras, so this would certainly be feasible. An 
encoder trained on millions of images could capture more detail than my simple network.

This model could easily be extended to identify other objects by adding more training data. In order to identify buildings,
for example, this model would only hav eto change the number of output classes. New training data with four classes 
would be necessary as well.