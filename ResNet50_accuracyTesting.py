import os
import zipfile
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import get_file

# Step 1: Load and prepare the Tiny ImageNet dataset
# Download Tiny ImageNet dataset (if not already downloaded)
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
dataset_path = get_file("tiny-imagenet-200.zip", url, untar=False)

# Unzip the Tiny ImageNet dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('/content/tiny-imagenet-200')

# Move the extracted files into a proper folder structure
# shutil.move('/content/tiny-imagenet-200/tiny-imagenet-200/', '/content/tiny-imagenet-200')

# Now Tiny ImageNet dataset is extracted and moved to the correct location

# Paths for the Tiny ImageNet validation set
val_dir = '/content/tiny-imagenet-200/tiny-imagenet-200/val/images'
annotations_file = '/content/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt'

# Verify that the validation image directory exists and has the correct structure
if not os.path.exists(val_dir):
    raise ValueError(f"Validation images directory does not exist at {val_dir}")
if len(os.listdir(val_dir)) == 0:
    raise ValueError(f"No images found in validation directory at {val_dir}")

# Load the annotations (image filenames and corresponding class labels)
with open(annotations_file, 'r') as f:
    lines = f.readlines()

# Create a dictionary for image filename to class label mapping
image_to_class = {}
for line in lines:
    parts = line.strip().split('\t')
    image_name, class_id = parts[0], parts[1]
    image_to_class[image_name] = class_id

# Initialize the ResNet50 model (using weights pre-trained on ImageNet)
model = ResNet50(weights='imagenet')

# Prepare the data generator for preprocessing
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Step 2: Process the validation images manually (flat structure)
subset_size = 50  # Process a subset of 50 images for faster results
image_names = os.listdir(val_dir)[:subset_size]  # Take first 'subset_size' images
images = []
labels = []

for image_name in image_names:
    # Load and preprocess the image
    img_path = os.path.join(val_dir, image_name)
    img = load_img(img_path, target_size=(224, 224))  # Resize for ResNet50
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    
    images.append(img_array)
    # Get the true label
    true_label = image_to_class[image_name]
    labels.append(true_label)

# Convert images to a numpy array
images = np.vstack(images)

# Step 3: Perform predictions with ResNet50 model
predictions = model.predict(images)

# Decode predictions into human-readable labels
decoded_predictions = decode_predictions(predictions, top=1)

# Debugging: Print out the predicted and true labels for a few samples
correct = 0
for i, pred in enumerate(decoded_predictions):
    # Get the true label from the image_to_class dictionary
    true_label = labels[i]
    
    # Get the predicted label from the decoded predictions (first label)
    predicted_label = pred[0][1]  # Top-1 prediction label (string)

    # Debugging: Print the true label and predicted label for comparison
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")

    # Compare predicted label with the true label (as both are class names)
    if true_label == predicted_label:
        correct += 1

# Calculate accuracy based on the number of correct predictions
accuracy = correct / len(image_names)
print(f'Accuracy on the subset: {accuracy * 100:.2f}%')
