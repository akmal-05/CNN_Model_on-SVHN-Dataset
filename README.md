# CNN Model On SVHN Dataset

1. Objective
The objective of this project is to train a Convolutional Neural Network (CNN) model to recognize and classify digits from the SVHN (Street View House Numbers) dataset. This dataset contains real-world images of house numbers taken from Google Street View, making it a challenging but practical dataset for digit recognition tasks.

2. Services the Model Provides:
- Digit Classification: The trained CNN model can accurately predict digits (0-9) in images taken from street views.
- Model Evaluation: It provides performance metrics such as accuracy, loss, and confusion matrix for assessing the model’s effectiveness.
- Real-World Application: The model is suitable for applications in autonomous vehicles and intelligent systems that require digit recognition in real-world scenarios.


3. Tools Used
- Programming Language: Python
Libraries and Frameworks:
- TensorFlow / Keras: For building, training, and testing the CNN model.
- NumPy: For efficient numerical computations.
- Pandas: For data manipulation and handling.
- Matplotlib: For visualizing training and evaluation metrics.
- scikit-learn: For additional tasks like train-test splitting and evaluation metrics.

4. CNN Model Architecture
- Input Layer: 32x32 RGB images (3 channels).
- Convolutional Layers: Multiple 2D convolutional layers for feature extraction.
- Max-Pooling Layers: Downsample feature maps to reduce dimensionality and retain important features.
- Fully Connected Layers: Flattened output connected to dense layers for classification.
- Output Layer: Softmax layer for digit classification (0-9).
5. Resources
Dataset: SVHN Dataset – A dataset containing over 600,000 labeled digits extracted from street images.
Documentation & Tutorials:
- TensorFlow/Keras Documentation – For building CNN models.
- Deep Learning with Python – A great book to understand CNN concepts.

Research Papers:
- Yu, T., et al. "SVHN: A Large-Scale Dataset for Street View House Numbers." (2016).


