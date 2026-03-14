# Laboratory-Work-3-Activity-Building-a-Custom-Image-Classifier

https://colab.research.google.com/drive/1P8zylblvjzJ7Px_p1LB1Qz4YOobPQsGk#scrollTo=u936Pl2x4RYw

# Laboratory Work 3 Activity — Building a Custom Image Classifier with TensorFlow Using Personal Image Datasets from Google Drive

Guide Questions (Student Reflection & Explanation)

Students must answer the following:

1. Dataset Preparation
   
○ How did you organize your dataset in Google Drive?
- I kept my data organized by creating a folder. In each folder, I created another folder with the name of each class or category I want to include in the model. Then, I put each image in its corresponding class folder. I made sure to name everything properly, with no spaces and in lower case, to prevent any path errors when loading the data in Colab.
  
○ Why is folder structure important for TensorFlow image loading?
- The literal meaning of TensorFlow's function image_dataset_from_directory() is to read your directory names as class labels. This function requires a very specific directory structure. So, the directory must contain subdirectories, and these subdirectories must contain the class. If your directories are named or placed incorrectly, TensorFlow either raises an error or, even worse, gets your class labels wrong. This directory structure is your class label system, and it must be correct from the beginning to save a tremendous amount of future debugging.

2. Model Training
     
○ What is the role of convolutional layers in image classification?
- Convolutional layers are the core of what makes a CNN actually "see." Instead of looking at an entire image at once, a convolutional layer scans through small regions of the image using filters and learns to detect patterns — things like edges, textures, curves, and color contrasts in the early layers, then more complex shapes like eyes or wheels in deeper layers. This is what allows the model to recognize an object regardless of where it appears in the image. Without convolutional layers, a regular neural network would just flatten the image into a list of pixel values, completely losing all spatial relationships.
  
○ Why do we split data into training and validation sets?
- The reason for this is that a model can "memorize" the images it is training on, as opposed to "learning" anything. If you have only tested your model on images it has also been training on, your accuracy figures would look much higher than they really are. The validation set is essentially a reality check, images that the model has not looked at before, so you can truthfully say whether it is really "learning" or "memorizing." It also helps you avoid "overfitting" your model, as you can tell when your training accuracy increases but your validation accuracy stops rising and may even decrease.
  
3. Performance Analysis
   
○ What accuracy did your model achieve?

- The model achieved a validation accuracy of around 99.9% after the completion of 15 to 20 epochs of training. The training accuracy was a little more, around 99%. This showed a minor level of overfitting, but nothing alarming. The model performed best for the classes for which there were the most images and the most visual diversity.
  
○ How did the number of images affect the model’s performance?
- It made a very noticeable difference. The classes where I had fewer than 200 images were consistently the classes that the model struggled with, either under-predicting them or getting them mixed up with classes that looked similar. The classes with 250+ images had much cleaner decision boundaries. Basically, the more images you can feed the model, the better it will understand what the actual features of a class are versus the accidental features that are present in the background. 

4. Critical Thinking
○ What challenges did you encounter while using your own dataset?
-Class imbalance — there were naturally more images for some categories than for others, and the model learned to favor the class with more images during prediction.
Image inconsistency — images taken from different angles, lighting, and distances made it difficult for the model to find a consistent pattern.

○ How can data augmentation improve your model?
  - Data augmentation is a technique to artificially increase your dataset by generating variations of your current images. This means you can turn an image around, rotate it slightly, or zoom in on it. This forces your model to be more robust because it cannot assume a certain orientation or lighting to classify an image. For a small custom dataset, this technique can almost double or triple your training images without acquiring a single new image. This directly helps you avoid overfitting because you are providing more images.
  - 
5. Application
○ Suggest a real-world application for your trained model.
- A good example of this is the application of a plant disease detection system for small-scale farmers. A machine learning model may be used to identify the condition of the leaves of the crops, whether they are in good condition or have already been affected by diseases or pests, based on the photo of the leaves. In the context of the geographical location of Mindanao, this may be useful to small-scale farmers.
  
○ How can this system be integrated into a mobile or web application?
- For a mobile application, the model can be converted to TensorFlow Lite (TFLite), a compressed and optimized version of the model, which can run directly on the Android or iOS device without the need for an internet connection. The user simply needs to open the application, point the camera at the crop leaf, and the model classifies the leaf immediately. 
For a web application, the model can be deployed as a REST API using a framework such as Flask or FastAPI, and the model runs as a cloud server, e.g., Google Cloud Run or Render. The user interface for the application, implemented using a framework such as React or simply HTML, enables the user to upload the image, and the application displays the prediction results. This approach is easier to maintain as the model runs as a server, and the application works for any device with a browser.

# Activity 3A: Improving and Evaluating a Custom Image Classifier

Guide Questions (Student Explanation & Reflection)
Students must answer:
Visualization & Overfitting
1. What signs indicated overfitting in your first model?
2. How did data augmentation affect validation accuracy?
Model Improvement
3. What is the purpose of dropout layers?
4. Why does data augmentation improve generalization?
Performance Comparison
5. Compare accuracy before and after improvements.
6. Which technique contributed most to improvement?

Deployment & Application
7. Why is saving the model important?
8. How can this model be deployed in a real-world system?
