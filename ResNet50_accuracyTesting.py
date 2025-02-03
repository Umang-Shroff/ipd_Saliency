import os
import zipfile
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import get_file

# Step 1: Load and prepare the Tiny ImageNet dataset
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
dataset_path = get_file("tiny-imagenet-200.zip", url, untar=False)

# Unzip the Tiny ImageNet dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('/content/tiny-imagenet-200')

# Paths for the Tiny ImageNet validation set
val_dir = '/content/tiny-imagenet-200/tiny-imagenet-200/val/images'
annotations_file = '/content/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt'

# Verify that the validation image directory exists
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

# Manually process the validation images (flat structure)
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

# Create a dictionary of ImageNet labels to Tiny ImageNet class IDs
imagenet_to_tinyimagenet = {
    'Kerry_blue_terrier': 'n01629819',
    'solar_dish': 'n02917067',
    'pomegranate': 'n07768694',
    'pitcher': 'n02909870',
    'corn': 'n04596742',
    'can_opener': 'n02988304',
    'consomme': 'n07920052',
    'eft': 'n01644900',
    'rock_beauty': 'n09256479',
    'sulphur_butterfly': 'n02190166',
    'washbasin': 'n07583066',
    'worm_fence': 'n03393912',
    'monastery': 'n03837869',
    'common_newt': 'n02106662',
    'Bouvier_des_Flandres': 'n02123045',
    'guinea_pig': 'n02364673',
    'banded_gecko': 'n02883205',
    'airship': 'n04023962',
    'rock_python': 'n06596364',
    'damselfly': 'n02268443',
    'caldron': 'n03444034',
    'space_heater': 'n04265275',
    'projectile': 'n02231487',
    'hare': 'n02002724',
    'vizsla': 'n02364673',
    'breastplate': 'n02963159',
    'alp': 'n03649909',
    'spotlight': 'n03706229',
    'candle': 'n02948072',
    'golf_ball': 'n07715103',
    'jigsaw_puzzle': 'n02730930',
    'mosque': 'n02699494',
    'container_ship': 'n02814533',
    'goldfish': 'n04356056',
    'yellow_lady\'s_slipper': 'n07753592',
    'weasel': 'n07695742',
    # You can add more mappings as needed
}

# Function to map ImageNet predictions to Tiny ImageNet classes
def map_imagenet_to_tinyimagenet(imagenet_label):
    return imagenet_to_tinyimagenet.get(imagenet_label, None)  # Return None if not found

# Perform predictions with ResNet50 model
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

    # Map the ImageNet predicted label to the Tiny ImageNet class ID
    mapped_predicted_label = map_imagenet_to_tinyimagenet(predicted_label)

    # Debugging: Print the true label, predicted label, and mapped predicted label
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}, Mapped Predicted Label: {mapped_predicted_label}")

    # Compare the mapped predicted label with the true label
    if mapped_predicted_label == true_label:
        correct += 1

# Calculate accuracy based on the number of correct predictions
accuracy = correct / len(image_names)
print(f'Accuracy on the subset: {accuracy * 100:.2f}%')
