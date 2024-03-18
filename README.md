<<<<<<< HEAD
# Sign Language Detection


<hr/>

## Dataset

The dataset used in this project consists of images of sign language gestures. You can use publicly available datasets like ASL Alphabet, ASL Finger Spelling, or create your own dataset.

<hr/>

## Requirements
- Python 3.10
- TensorFlow


<hr/>

## Usage
1. **Data Preparation**: Preprocess the dataset by standardizing the size, color, and orientation of the images. Also, label the data with the corresponding sign language gestures.

2. **Model Architecture**: Design the CNN architecture. This repository provides a simple example architecture using TensorFlow/Keras.

3. **Training**: Train the model on the preprocessed dataset using the provided architecture. Adjust hyperparameters as necessary to improve performance.

4. **Evaluation**: Evaluate the trained model on a separate test set to assess its performance metrics such as accuracy, precision, recall, and F1-score.

5. **Deployment**: Deploy the trained model to a platform where it can be used for sign language detection, such as a web application, mobile app, or embedded device.

<hr/>


## Screenshot

![App Screenshot](https://res.cloudinary.com/dwdntz8et/image/upload/v1710599975/SignLanguage_gr5or4.png)

<hr/>

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Python](https://www.python.org) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/pradeep8577/Sign-Language.git

# Creation of virtual environments
$ pip install virtualenv

# Create It
$ virtualenv env

# Activate It
$ env/scripts/activate

# Install dependencies
$ pip install -r requirements.txt

# Run the application
$ streamlit run app.py

# Visit the following URL in your web browser
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.5:8501
```

<hr/>
<hr/>

## Object Detection Pipeline using YOLOv8 Nano

This Python code provides a pipeline for detecting signs in images using the YOLOv8 Nano model. It consists of several components:

1. **Model Initialization**: The YOLOv8 Nano model is loaded using the `ultralytics` library.

2. **Data Preprocessing**: Images are preprocessed by converting them to RGB format and converting them into numpy arrays.

3. **Object Detection**: The YOLO model is used to detect objects in the images. Detected objects are filtered to extract sign detections, which include bounding box coordinates, confidence scores, and class IDs.

4. **Drawing Detections on Images**: Detected signs are visualized on the input images by drawing bounding boxes around them. Class names and confidence scores are also displayed near the bounding boxes.

5. **Text Results Extraction**: The detected signs and their corresponding confidence scores are extracted as text results.

## Usage:

To use this pipeline:

1. **Instantiate the Pipeline**: Create an instance of the `detectPipeline` class.

2. **Detect Signs in an Image**: Use the `detect_signs` method by passing the path of the image you want to analyze. This method returns a list of sign detections.

3. **Draw Detections on Image**: Use the `drawDetections2Image` method to draw bounding boxes and labels on the original image. This method returns the image with detections visualized.

4. **Extract Text Results**: Use the `extractTextResults` method to obtain the text representation of the detected signs along with their confidence scores.

## Dependencies:

- **Ultralytics**: Install Ultralytics library for object detection models.
- **Pillow**: Install Pillow for image processing.
- **NumPy**: Install NumPy for numerical operations.
- **OpenCV (cv2)**: Install OpenCV for image manipulation and visualization.
- **Pandas**: Install Pandas for data manipulation and analysis.

## File Structure:

- **Models/yolo_v8_nano_model.pt**: Pre-trained YOLOv8 Nano model.
- **detectPipeline.py**: Python file containing the object detection pipeline class.

## Class Methods:

- **`detect_signs(img_path: str)`:** Detects signs in an image and returns a list of sign detections.
- **`drawDetections2Image(img_path, detections)`:** Draws bounding boxes and labels on the original image and returns the modified image.
- **`extractTextResults(detections)`:** Extracts text results (sign labels and confidence scores) from the detected signs.

## Example Usage:

```python
from detectPipeline import detectPipeline

pipeline = detectPipeline()

# Detect signs in an image
detections = pipeline.detect_signs('path/to/your/image.jpg')

# Draw detections on the original image
image_with_detections = pipeline.drawDetections2Image('path/to/your/image.jpg', detections)

# Extract text results
text_results = pipeline.extractTextResults(detections)

print(text_results)

# Display or save the image with detections
cv.imshow('Image with Detections', image_with_detections)
cv.waitKey(0)

```
<hr/>

## Convolutional neural network (CNN) 

This code is defining a convolutional neural network (CNN) model using the Keras framework with a TensorFlow backend. Let's break it down line by line:

```python
model = Sequential()
```
This line initializes a Sequential model, which is a linear stack of layers. Layers can be added to this model one by one, and each layer passes its output to the next layer in the stack.

```python
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
```
This line adds a 2D convolutional layer (`Conv2D`) to the model. It has 75 filters with a kernel size of 3x3. `strides=1` means the filter moves one pixel at a time. `padding='same'` means the input is zero-padded so that the output has the same spatial dimensions as the input. `activation='relu'` sets the activation function to ReLU (Rectified Linear Unit). `input_shape=(28,28,1)` specifies the shape of the input data: 28x28 pixels with a single channel (grayscale).

```python
model.add(BatchNormalization())
```
This line adds a batch normalization layer. Batch normalization normalizes the activations of the previous layer at each batch.

```python
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
```
This line adds a max-pooling layer (`MaxPool2D`) with a pool size of 2x2, a stride of 2 (which means the pooling window moves 2 pixels at a time), and 'same' padding, which pads the input if necessary to ensure that the output has the same height and width as the input.

```python
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
```
This line adds another convolutional layer similar to the first one but with 50 filters instead of 75.

```python
model.add(Dropout(0.2))
```
This line adds a dropout layer. Dropout is a regularization technique that randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting.

```python
model.add(BatchNormalization())
```
Another batch normalization layer is added after dropout.

```python
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
```
Another max-pooling layer is added.

```python
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
```
Another convolutional layer is added, this time with 25 filters.

```python
model.add(BatchNormalization())
```
Another batch normalization layer is added.

```python
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
```
Another max-pooling layer is added.

```python
model.add(Flatten())
```
This line adds a flatten layer, which flattens the input into a one-dimensional array. This is necessary before passing the data to fully connected layers.

```python
model.add(Dense(units = 512 , activation = 'relu'))
```
This line adds a fully connected (dense) layer with 512 units and ReLU activation function.

```python
model.add(Dropout(0.3))
```
Another dropout layer is added.

```python
model.add(Dense(units = 26 , activation = 'softmax'))
```
Finally, a dense layer with 26 units (assuming it's a classification task with 26 classes, perhaps for recognizing characters) and softmax activation function is added. Softmax is often used in the output layer of a classification model to convert the model's raw output into probabilities.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
This line compiles the model, configuring it for training. It specifies the loss function (`categorical_crossentropy` for multi-class classification), the optimizer (`adam`), and the evaluation metric (`accuracy`).

<hr/>
=======
# Sign-language-detection
>>>>>>> 1e2a8a5bfe32604a34d902ae24811952c1b13520
