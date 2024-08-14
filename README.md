# IC-BGAN
 A Novel Interpolation Consistency for Bad Generative Adversarial Networks (IC-BGANs) 

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

Semi-supervised learning techniques utilize both labeled and unlabeled images to enhance classification performance in scenarios where labeled images are limited. However, challenges such as integrating unlabeled images with incorrect pseudo-labels, determining appropriate thresholds for the pseudo-labels, and label prediction fluctuations on low-confidence unlabeled images, hinder the effectiveness of existing methods. This research introduces a novel framework named Interpolation Consistency for Bad Generative Adversarial Networks (IC-BGAN) that utilizes a new loss function. The proposed model combines bad adversarial training, fusion techniques, and regularization to address the limitations of semi-supervised learning. IC-BGAN creates three types of image augmentations and label consistency regularization in interpolation of bad fake images, real and bad fake images, and unlabeled images. It demonstrates linear interpolation behavior, reducing fluctuations in predictions, improving stability, and facilitating the identification of decision boundaries in low-density areas. The regularization techniques boost the discriminative capability of the classifier and discriminator, and send a better signal to the bad generator. This improves the generalization and the generation of diverse inter-class fake images as support vectors with information near the true decision boundary, which helps to correct the pseudo-labeling of unlabeled images. The proposed approach achieves notable improvements in error rate from 2.87 to 1.47 on the Modified National Institute of Standards and Technology (MNIST) dataset, 3.59 to 3.13 on the Street View House Numbers (SVHN) dataset, and 12.13 to 9.59 on the Canadian Institute for Advanced Research, 10 classes (CIFAR-10) dataset using 1000 labeled training images. Additionally, it reduces the error rate from 22.11 to 18.40 on the CINIC-10 dataset when using 700 labeled images per class. The experiments demonstrate the IC-BGAN framework outperforms existing semi-supervised methods, providing a more accurate classification solution with smoother class label estimates, especially for low-confidence unlabeled images.
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
