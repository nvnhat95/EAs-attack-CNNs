# EAs-attack-CNNs
perform adversarial attacks on CNNs using Evolutionary Algorithms
This project includes 2 different types of attacking:
  - all pixel attack: perform on inception-v3 integrated in keras with ImageNet 2012 dataset. Attack methods: GA and LS-CMA-ES which is a novel method allowing the use of CMA-ES in high dimentional spaces. See the paper for more details.
  - 1 pixel (or several pixels) attack: perform on a resnet model with CIFAR-10 dataset. Attack methods: GA, CMA-ES and DE.
    The model is taken from here: https://github.com/Hyperparticle/one-pixel-attack-keras/tree/master/networks/models?fbclid=IwAR1PnqVZxwGManLSaGHflJ5bEe7Tg_a_f_y2LqIRelaCfBPyuNm1-84aRqk

This project takes into to account that values of pixels in the image must be intergers. The results are promising.

For the use of attackings, please refer to notebook files
