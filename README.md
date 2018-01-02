# Noise robustness of neural network and SVM classifiers
The data is for a 2-dimensional 4-class problem. (The class labels are 0–3). Different files have different level of noise in labels. We need to train classifiers using the noisy data and test it on clean data.
## Implementation Details :
1. Neural network
  * Number of neurons in input layer: 2
    Number of neurons in input layer: 8
  * Activation function
    – Hidden layer: sigmoid activation function
    – Output layer: softmax activation function
      ~ used to turn the outputs into probability-like values and allow one class of the 4 to be selected as the model’s output prediction.
  * Loss function: Logarithmic loss
  * ADAM gradient descent algorithm is used to learn the weights.
2. SVM
  * kernel: Gaussian kernel (RBF)
  * C=1000
  * Note: SVM with the polynomial kernel (degree = 2/3/4, C = 10/100/500/1000) had low accuracy on clean data and hence was not used.
3. Each file is divided randomly into training (80%) and test set(20%). Models are trained on each training set and accuracy is calculated on the noisy and clean test set as shown in tables below.

![](https://i.imgur.com/BOLTfAB.jpg)

## Conclusions:
  * From clean test data accuracies, in above tables, we can infer that neural network is comparatively more robust to noise as compared to SVM.
  * Models trained on noisy training data (with noisy training data accuracy ≈ 0.40) gains an accuracy of more than 0.90 on both SVM and neural networks for clean test data.
  * Both neural networks and SVM show very good robustness to noise.
