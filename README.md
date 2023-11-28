# ICBSGAN
A Novel Interpolation Consistency for Bad Semi-Supervised Generative Adversarial Networks (ICBSGAN) in Image Classification and Interperation 

## Table of Contents
- [Authors](#authors)
- [Abstract](#abstract)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Authors

- Mohammad Saber Iraji
- Jafar Tanha (Corresponding author: tanha@tabrizu.ac.ir, jafar.tanha.pnu@gmail.com)
- Mohammad-Ali Balafar
- Mohammad-Reza Feizi-Derakhshi

This work was conducted by researchers from the Department of Computer Engineering, Faculty of Electrical and Computer Engineering, University of Tabriz, Tabriz, Iran.

## Abstract

Semi-supervised learning techniques leverage both labeled and unlabeled images to enhance classification performance in scenarios where labeled images are limited. However, challenges, such as determining appropriate thresholds, integrating incorrect pseudo-labels, and establishing effective consistency in augmentations, hinder the effectiveness of existing methods. Additionally, label prediction fluctuations on low-confidence images and their impact on generalization performance pose further limitations. This research introduces a novel framework, named interpolation consistency for bad semi-supervised generative adversarial networks (ICBSGAN) which addresses the limitations of semi-supervised learning through a new loss function. The proposed model combines bad adversarial training, fusion techniques, and regularization to enhance semi-supervised learning. ICBSGAN incorporates three types of interpolation consistency training: interpolation of bad fake images, real and bad fake images, and unlabeled images. The regularization techniques improve the generalization and the generation of diverse fake images as support vectors in low-density areas. It demonstrates linear behavior at interpolation, reducing fluctuations in predictions, improving stability, and the identification of decision boundaries. Experimental evaluations on the CIFAR-10, CINIC-10, MNIST, and SVHN datasets showcase the effectiveness of ICBSGAN compared to the state-of-the-art methods. The proposed approach achieves notable improvements in error rate from 2.87 to 1.47 on MNIST, 3.89 to 3.13 on SVHN, and 15.48 to 9.59 on CIFAR-10 using 1000 labeled training images. Additionally, it reduces the error rate from 22.11 to 18.40 on CINIC-10 when using 700 labeled images per class. The code can be found at the following GitHub repository: https://github.com/ms-iraji/ICBSGAN ↗.
## Key Features
Informative fake images, 
Low-confidence images, 
Interpolation consistency regularization, 
Semi-supervised learning, 
Bad adversarial training, 
Image fusion

## Installation and Usage

To use the ICBSGAN algorithm, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Configure the training parameters and dataset paths in the provided configuration file.
4. Evaluate the trained model using `mainmnist.py`.


## Results

 Experimental evaluations on the CIFAR-10, CINIC-10, MNIST, and SVHN datasets showcase the effectiveness of ICBSGAN compared to the state-of-the-art methods. The proposed approach achieves notable improvements in error rate from 2.87 to 1.47 on MNIST, 3.89 to 3.13 on SVHN, and 15.48 to 9.59 on CIFAR-10 using 1000 labeled training images. Additionally, it reduces the error rate from 22.11 to 18.40 on CINIC-10 when using 700 labeled images per class.  For detailed results, please refer to the [Results](#results) section in the paper.


## Contributing

Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request or open an issue on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
