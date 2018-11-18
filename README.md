# Semantic Segmentation Architectures

## Description
This repository contais the implementation of Semantic Segmentation Deep Learning architectures
for the Computer Vision course. For more info about the course, please refer to its [website (in Portuguese)](http://www.cin.ufpe.br/~cabm/visao/).

## Models
The following segmentation models are currently made available:

- [Encoder-Decoder based on SegNet](https://arxiv.org/abs/1511.00561). This fully convolutional network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions.

- [Encoder-Decoder with skip connections based on SegNet](https://arxiv.org/abs/1511.00561). This fully convolutional network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions.
Also, it employs additive skip connections from the encoder to the decoder. 

## Dataset
- The class segmentation: pixel indices correspond to classes in alphabetical order (1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor). The index 0 corresponds to background.

## Files and Directories 

## Results

- [Encoder-Decoder based on SegNet](https://arxiv.org/abs/1511.00561)

<p float="left">
<img height="275" width='275' src="results/input/2007_000033.jpg">
<img height='275' src="results/2007_000033.png">
<img height="275" width='275' src="results/gt/2007_000033.png">
<p>

<p float="left">
<img height="275" width='275' src="results/input/2007_000039.jpg">
<img height='275' src="results/2007_000039.png">
<img height="275" width='275' src="results/gt/2007_000039.png">
<p>

<p float="left">
<img height="275" width='275' src="results/input/2007_000042.jpg">
<img height='275' src="results/2007_000042.png">
<img height="275" width='275' src="results/gt/2007_000042.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_000663.jpg">
<img height='275' src="results/2007_000663.png">
<img height="275" width='275' src="results/gt/2007_000663.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_000762.jpg">
<img height='275' src="results/2007_000762.png">
<img height="275" width='275' src="results/gt/2007_000762.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_000799.jpg">
<img height='275' src="results/2007_000799.png">
<img height="275" width='275' src="results/gt/2007_000799.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_004712.jpg">
<img height='275' src="results/2007_004712.png">
<img height="275" width='275' src="results/gt/2007_004712.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_005331.jpg">
<img height='275' src="results/2007_005331.png">
<img height="275" width='275' src="results/gt/2007_005331.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_005626.jpg">
<img height='275' src="results/2007_005626.png">
<img height="275" width='275' src="results/gt/2007_005626.png">
</p>

<p float="left">
<img height="275" width='275' src="results/input/2007_009251.jpg">
<img height='275' src="results/2007_009251.png">
<img height="275" width='275' src="results/gt/2007_009251.png">
</p>

- [Encoder-Decoder with skip connections based on SegNet](https://arxiv.org/abs/1511.00561).