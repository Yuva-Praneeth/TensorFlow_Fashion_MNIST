# Fashion MNIST Image Classification (Deep Learning Term Project)

## Project Overview
- Deep learning model to classify fashion items from the Fashion MNIST dataset
- Built with TensorFlow 2.x and Keras
- Convolutional Neural Network (CNN) architecture
- 10 fashion categories: T-shirts, Trousers, Pullovers, Dresses, Coats, Sandals, Shirts, Sneakers, Bags, Ankle boots

## Technical Details
- Two convolutional layers with 64 filters (3×3 kernel)
- MaxPooling with 2×2 pool size
- Dropout rate of 0.2 after each pooling layer
- Actiavtion : Relu
- Final dense layer : softmax activation for multi-class classification
- Adam optimizer with learning rate 1e-3
- Loss function: sparse categorical crossentropy
- Images normalized to [0,1] range
- Training conducted for 2 epochs with batch size of 32
- Model evaluation using confusion matrix visualization

## Performance Metrics
- Accuracy score calculation
- Precision score (weighted average)
- Recall score (weighted average)
- F1 score (weighted average)

## Data Handling
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28×28 pixels (grayscale)

## Required Libraries
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Getting Started
- Run the script to train the model and evaluate results
- The code automatically downloads and preprocesses the Fashion MNIST dataset
- Results include confusion matrix and prediction visualizations
