# Edge-Detection-Kernel-Creator
Convolutional Neural Network that figures out the kernel you give it according to the image output.

Theoretical Implementation of Computerphile's Kernel Convolution video: 
https://www.youtube.com/watch?v=C_zFhWdM4ic

This page started with me re-inventing the wheel, this time for Image Kernel Convolutions. Instead of the usual approach of squeezing your image into a vector and composing convolution / cross-correlation for a given Kernel like Tensorflow or Open-CV do, this one does the following:

For a 3x3 Kernel, create 9 copies of your image as Numpy arrays and adjust each accordingly by cutting off 2 columns/rows (since for a 3x3 kernel out output will lose 2 pixels) , multiplying each of the 9 by its appropriate ocnvolution filter scalar, and adding it together, so instead of the convolution process going row by row with Quadratic time, its one linear operation. 

Now we have some high level functions that do basic edge detection. Let's step it up a notch. 

Given one convolutional Kernel, let's say for example the vertical edge detection:
[[-1,-2,-1],
[0,0,0]
[1,2,1]]
And apply it to an image. Now we can build a neural network that does a convolution with randomly initialized weights and compares is to our given output. The loss is described as the per pixel value loss, and backporpagation allows us to find Kernel weights that give identical results. 

The most fascinating aspect of this project is that the results the neural net gives out various filters that all give the same, or similar output. This is an example of machine intelligence teaching humans. 

![alt tag](https://github.com/ConsciousMachines/Edge-Detection-Kernel-Creator/blob/master/Kernel%20Creator%20ex1.gif)
