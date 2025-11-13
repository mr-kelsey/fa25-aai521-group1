# Group 1 final project for AAI-521

This project is a part of the AAI-521 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: [Planned, **Active**, On-hold, Completed]

#### Installation
You should add an instruction how this project is to be used, installed, run, edited in others’ machine.
 
#### Project Intro/Objective

The main purpose of this project is to provide a web based user interface that allows a user to upload a damaged photo for restoration.

#### Partner(s)/Contributor(s)

* Johnathan Kelsey
* [https://github.com/mr-kelsey](https://github.com/mr-kelsey)
* Gregory Bauer
* [https://github.com/gbauer-at-sandiego-edu](https://github.com/gbauer-at-sandiego-edu)
* Atul Aneja
* [https://github.com/AtulAneja](https://github.com/AtulAneja)

#### Methods Used
* Computer Vision
* Machine Learning
* Deep Learning
* Data Visualization
* Cloud Computing 
* Data Manipulation

#### Technologies
* Python

#### Project Description
This project intends to manipulate photos in various ways.  It will be able to fill in missing portions of an image, reduce noise, create a higher resolution image and color in photos taken in black and white.  We will rely on several hugging face models to achieve these capabilities.

* Super-resolution
 * Model: CompVis/ldm-super-resolution-4x-openimages
 * URL: https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages
* Denoising
 * Model: google/ddpm-cifar10-32
 * URL: https://huggingface.co/google/ddpm-cifar10-32
* Colorization
 * Model: microsoft/ColorizationTransformer
 * URL: https://huggingface.co/microsoft/ColorizationTransformer
* Inpainting
 * Model: stabilityai/stable-diffusion-2-inpainting
 * URL: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

We will use a sample of the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset for our fine tuning.  The enitre dataset is comprised of 900 high-res images (2040 x M).  As such, the dataset is nearly 4Gb and is too large to use in it's entirety.  As such, we opted to keep 56 of the images and use those.

#### License
GNU GENERAL PUBLIC LICENSE Version 3

#### Acknowledgments
You can mention and thank your professors and those who technically helped you during the project. 
