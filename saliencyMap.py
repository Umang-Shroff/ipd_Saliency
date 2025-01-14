# CODE SNIPPET 1
# FOR RUNNING IN COLAB USE ! BEFORE INSTALL
!pip install keras-vis
!pip install tensorflow
!pip install tf-keras-vis tensorflow
!pip install numpy pillow

# FOR CODE EDITORS
# IN TERMINAL
pip install keras-vis
pip install tensorflow
pip install tf-keras-vis tensorflow
pip install numpy pillow

# FOR IMPORTING IMAGE FROM GOOGLE DRIVE
#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext autoreload
# %autoreload 2

import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input

import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections.abc import Iterable
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils
from tensorflow.keras.applications.vgg16 import decode_predictions
import json








# CODE SNIPPET 2
def display_imgs(images):
  subprot_args = {
   'nrows': 1,
   'ncols': 4,
   'figsize': (6, 3),
   'subplot_kw': {'xticks': [], 'yticks': []}
  }
  f, ax = plt.subplots(**subprot_args)
  for i in range(len(images)):
    ax[i].imshow(images[i])
  plt.tight_layout()
  plt.show()

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()

#Load images
#img1 = load_img('/content/drive/MyDrive/test_imgs/cat.jpg', target_size=(224, 224))
#img2 = load_img('/content/drive/MyDrive/test_imgs/dog.jpg', target_size=(224, 224))
#img3 = load_img('/content/drive/MyDrive/test_imgs/hen.jpg', target_size=(224, 224))
#img4 = load_img('/content/drive/MyDrive/test_imgs/tiger.jpeg', target_size=(224, 224))


# ENTER THE NAME OF THE IMAGE UPLOADED IN THE ASSETS
img1 = load_img('Tiger_img.jpg', target_size=(224, 224))
img2 = load_img('car_img.jpg', target_size=(224, 224))
img3 = load_img('plane_img.jpg', target_size=(224, 224))
img4 = load_img('lotus_img.jpg', target_size=(224, 224))

#plt.imshow(img1)
#plt.show()
#plt.imshow(img2)
#plt.show()
#plt.imshow(img3)
#plt.show()
#plt.imshow(img4)
#plt.show()

#create array of images
images = np.asarray([np.array(img1), np.array(img2), np.array(img3), np.array(img4)])

#show images
display_imgs(images)









# CODE SNIPPET 3
#convert to numpy array for reshaping
img1 = img_to_array(img1)
img2 = img_to_array(img2)
img3 = img_to_array(img3)
img4 = img_to_array(img4)

#reshape to prepare for processing
img1 = img1.reshape(1,224,224,3)
img2 = img2.reshape(1,224,224,3)
img3 = img3.reshape(1,224,224,3)
img4 = img4.reshape(1,224,224,3)

#preprocess to prepare for input
img1 = preprocess_input(img1)
img2 = preprocess_input(img2)
img3 = preprocess_input(img3)
img4 = preprocess_input(img4)

# predictions with input images
yhat1 = model.predict(img1)
yhat2 = model.predict(img2)
yhat3 = model.predict(img3)
yhat4 = model.predict(img4)

#decode predictions
label1 = decode_predictions(yhat1)
label2 = decode_predictions(yhat2)
label3 = decode_predictions(yhat3)
label4 = decode_predictions(yhat4)

# extract top most prediction for each input
label1 = label1[0][0]
label2 = label2[0][0]
label3 = label3[0][0]
label4 = label4[0][0]

#plt.imshow(image1)
print('%s (%.2f%%)' % (label1[1], label1[2]*100))
#plt.imshow(image2)
print('%s (%.2f%%)' % (label2[1], label2[2]*100))
#plt.imshow(image3)
print('%s (%.2f%%)' % (label3[1], label3[2]*100))
#plt.imshow(image4)
print('%s (%.2f%%)' % (label4[1], label4[2]*100))








# CODE SNIPPET 4
#prepare 1000 classes
CLASS_INDEX = json.load(open("imagenet_class_index.json"))
classlabel = []
for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])
print("N of class={}".format(len(classlabel)))

#Top 5 classes predicted
class_idxs_sorted1 = np.argsort(yhat1.flatten())[::-1]
class_idxs_sorted2 = np.argsort(yhat2.flatten())[::-1]
class_idxs_sorted3 = np.argsort(yhat3.flatten())[::-1]
class_idxs_sorted4 = np.argsort(yhat4.flatten())[::-1]

topNclass = 5

print('\nfirst image\n')
for i, idx in enumerate(class_idxs_sorted1[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat1[0,idx]))

print('\nsecond image\n')
for i, idx in enumerate(class_idxs_sorted2[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat2[0,idx]))

print('\nthird image\n')
for i, idx in enumerate(class_idxs_sorted3[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat3[0,idx]))

print('\nFourth image\n')
for i, idx in enumerate(class_idxs_sorted4[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat4[0,idx]))
    
    
    
    

# CODE SNIPPET 5
# swap softmax layer with linear layer
layer_idx = utils.find_layer_idx(model, 'predictions')
model.layers[-1].activation = tf.keras.activations.linear
model = utils.apply_modifications(model)

#get the input image index
from tf_keras_vis.utils.scores import CategoricalScore
#cat - 281, dog -235 , hen -8, tiger - 292
score = CategoricalScore([281, 235, 8 , 292])

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# ENTER THE NAME OF THE IMAGES UPLOADED FOR LABELLING
input_classes = ['Tiger', 'Car', 'Plane', 'Lotus']

input_images = preprocess_input(images)

# Create Gradcam object
gradcam = Gradcam(model,
                  clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score,
              input_images,
              penultimate_layer=-1)

#show generated images
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, img_class in enumerate(input_classes):
    heatmap = np.uint8(cm.jet(cam[i])[..., :4] * 255)
    ax[i].set_title(img_class, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
    ax[i].axis('off')
plt.tight_layout()
plt.show()






# CODE SNIPPET 6
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

#Create Saliency object
saliency = Saliency(model, clone=False)

# Generate saliency map
saliency_map = saliency(score, input_images)
saliency_map = normalize(saliency_map)

subprot_args = {
   'nrows': 1,
   'ncols': 4,
   'figsize': (6, 3),
   'subplot_kw': {'xticks': [], 'yticks': []}
}
f, ax = plt.subplots(**subprot_args)
for i in range(len(saliency_map)):
    # THE CMAP PROPERTY IS THE GRADENT IN WHICH IMAGE IS SHOWN
    # TAKE VALUES  gray_r / binary / gist_yarg
   ax[i].imshow(saliency_map[i], cmap='gray_r')
plt.tight_layout()
plt.show()

saliency_map = saliency(score, input_images, smooth_samples=20)
saliency_map = normalize(saliency_map)

f, ax = plt.subplots(**subprot_args)
for i in range(len(saliency_map)):
    # THE CMAP PROPERTY IS THE GRADENT IN WHICH IMAGE IS SHOWN
    # TAKE VALUES  gray_r / binary / gist_yarg
   ax[i].imshow(saliency_map[i], cmap='gray_r')
plt.tight_layout()
plt.show()







# CODE SNIPPET 7
# FOR SAVING THE GENERATED IMAGES IN THE ASSETS FOLDER
def save_saliency_map(saliency_map, file_name):
    for i in range(len(saliency_map)):
        plt.imsave(f'{file_name}_{i}.png', saliency_map[i], cmap='gray_r', format='png')
save_saliency_map(saliency_map, 'saliency_map')






# CODE SNIPPET 8
# FOR CONVERTING SALIENCY MAP TO AN ARRAY/MATRIX AND CONVERTING IT BACK TO IMG
from PIL import Image
import numpy as np

def image_to_matrix(image_path):
    pil_image = Image.open(image_path).convert('RGBA')  # Ensure image is in RGBA format
    image_matrix = np.array(pil_image)
    return image_matrix

# ENTER THE IMAGE FILE NAME WHICH HAS TO BE CONVERTED TO A MATRIX
image_path = 'saliency_map_1.png' 
matrix = image_to_matrix(image_path)

print("Matrix shape:", matrix.shape)
print("Matrix data (sample):")
print(matrix)

def matrix_to_image(matrix, output_path):
    pil_image = Image.fromarray(matrix, 'RGBA')  # Ensure matrix is in RGBA format
    pil_image.save(output_path)

# ENTER THE NAME WITH WHICH THE OUTPUT IMAGE SHOULD BE SAVED AS
output_path = 'OUTPUT_IMAGE.png'
matrix_to_image(matrix, output_path)
print(f"Image saved to {output_path}")
