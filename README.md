# Image Segmentation with DeepLabV3
This project demonstrates how to perform image segmentation using pre-trained DeepLabV3 models with ResNet backbones. The script uses PyTorch and OpenCV for processing and visualizing images.

# Approach
The script takes an image as input, applies a pre-trained DeepLabV3 model to perform segmentation, and visualizes the segmented image. The model identifies and labels objects within the image using a predefined label map.

# Libraries Used
**torch**: For loading and running the deep learning model.
**torchvision**: For accessing pre-trained models and their weights.
**torchvision.models.segmentation**: For DeepLabV3 segmentation models with ResNet backbones.
**deeplabv3_resnet50**: DeepLabV3 model with a ResNet-50 backbone.
**deeplabv3_resnet101**: DeepLabV3 model with a ResNet-101 backbone.
**DeepLabV3_ResNet50_Weights**: Pre-trained weights for the ResNet-50 model.
**DeepLabV3_ResNet101_Weights**: Pre-trained weights for the ResNet-101 model.
**opencv-python-headless**: For image processing tasks.
**matplotlib**: For visualizing images and segmentation results.
**PIL**: For image handling and manipulation.
**numpy**: For numerical operations and array manipulations.
**google.colab**: For handling file uploads and downloads in Google Colab.

# Output Format
The output consists of:
- **Original Image**: The uploaded image displayed in its original form.
- **Segmented Image**: The output of the segmentation, colored according to the defined label map, displayed alongside the original image.

## Usage
1. Run the script in a Google Colab environment.
2. Upload an image when prompted.
3. The script will perform segmentation using a pre-trained DeepLabV3 model and display the original and segmented images.

## Considerations
- **Model Selection**: When using a pre-trained segmentation model like DeepLabV3, you can choose between different backbone architectures such as ResNet-50 and ResNet-101. The choice of model can affect the accuracy and speed of the segmentation.In this case,the script allows switching between ResNet-50 and ResNet-101 models, providing flexibility to balance accuracy and computational resource requirements.
- **Computational Resource Requirements**: DeepLabV3 models are computationally intensive and may require significant resources, especially for large images. Running these models on a GPU can significantly speed up the inference process. Ensure that you have access to a suitable computational environment, such as Google Colab with GPU support.In this case,it automatically uses a GPU if available, significantly speeding up the inference process. For environments without a GPU, the script defaults to CPU, ensuring broader applicability.

# Here is the Python Code:
# Install required libraries
!pip install torch torchvision opencv-python-headless matplotlib

"""Import necessary libraries"""

# Import the OS library for operating system dependent functionality
import os

# Import OpenCV for image processing tasks
import cv2

# Import the PIL (Pillow) library for image handling and manipulation
import PIL

# Import PyTorch for deep learning operations and model handling
import torch

# Import NumPy for numerical operations and array manipulations
import numpy as np

# Import Matplotlib for plotting and visualizing images
import matplotlib.pyplot as plt

# Import the files module from Google Colab for handling file uploads and downloads
from google.colab import files

# Import the DeepLabV3 models with ResNet architectures for semantic segmentation
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

# Import pre-trained weights for the DeepLabV3 models
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,  # Weights for the ResNet-50 model
    DeepLabV3_ResNet101_Weights   # Weights for the ResNet-101 model
)

"""Define label map"""

# Define label map
label_map = np.array([
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128),  # tv/monitor
])

"""Define helper functions"""

# Define helper functions for drawing segmentation map
def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).numpy()  # Get predicted labels
    red_map = np.zeros_like(labels).astype(np.uint8)  # Initialize red channel
    green_map = np.zeros_like(labels).astype(np.uint8)  # Initialize green channel
    blue_map = np.zeros_like(labels).astype(np.uint8)  # Initialize blue channel

    # Create RGB segmentation map based on label map
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        R, G, B = label_map[label_num]
        red_map[index] = R
        green_map[index] = G
        blue_map[index] = B

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)  # Stack channels
    return segmentation_map  # Return the segmentation map

"""Load model function"""

"""
    Load a pre-trained DeepLabV3 model.

    Args:
    model_name (str): The name of the model to load ('resnet_50' or 'resnet_101').

    Returns:
    model: The loaded model.
    transforms: The transformation applied to input images.
"""

def load_model(model_name: str):
    if model_name.lower() not in ("resnet_50", "resnet_101"):
        raise ValueError("'model_name' should be one of ('mobilenet', 'resnet_50', 'resnet_101')")

     # Load the specified model and its associated weights
    if model_name == "resnet_50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    else:
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    model.eval()  # Set the model to evaluation mode
    _ = model(torch.randn(1, 3, 520, 520))  # Run a dummy input through the model to load weights
    return model, transforms  # Return the model and transformations

"""Perform inference function"""

"""
    Perform image segmentation using the specified model.

    Args:
    model_name (str): The name of the model to use.
    image_path (str): The path to the input image.
    device (str): The device to run the model on ('cuda' or 'cpu').
"""
def perform_inference(model_name: str, image_path=None, device=None):
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model, transforms = load_model(model_name)  # Load the model and transforms
    model.to(device)    # Move model to the specified device

    img_raw = PIL.Image.open(image_path).convert("RGB")  # Load and convert image to RGB
    W, H = img_raw.size[:2]  # Get image dimensions
    img_t = transforms(img_raw)  # Apply transformations
    img_t = torch.unsqueeze(img_t, dim=0).to(device)  # Add batch dimension and move to device


    with torch.no_grad():  # Disable gradient computation
        output = model(img_t)["out"].cpu()  # Get model output

    segmented_image = draw_segmentation_map(output)  # Create segmentation map
    segmented_image = cv2.resize(segmented_image, (W, H), cv2.INTER_LINEAR)  # Resize to original dimensions

    # Display original image and segmentation result
    plt.figure(figsize=(12, 10), dpi=100)
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Image")
    plt.imshow(np.asarray(img_raw))

    plt.subplot(1, 3, 2)
    plt.title("Segmentation")
    plt.axis("off")
    plt.imshow(segmented_image)


    plt.show()
    plt.close()

"""Download and extract Pascal VOC 2012 dataset"""

# Step 7: Download and extract Pascal VOC 2012 dataset
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf VOCtrainval_11-May-2012.tar

"""Upload an image and perform inference"""

"""
    Upload an image and perform segmentation.

    Args:
    model_name (str): The name of the model to use for segmentation.
"""
def upload_and_segment_image(model_name='resnet_50'):
    uploaded = files.upload()  # Upload the image
    for filename in uploaded.keys():  # Process each uploaded file
        perform_inference(model_name=model_name, image_path=filename)  # Perform segmentation

"""Upload an image for segmentation"""

# Step 9: Upload an image and segment it
upload_and_segment_image('resnet_101') # Perform segmentation using ResNet101


## Example
After running the script with an example image, you will see:
- The pixel data processed and the original image displayed.
- The segmentation results visualized alongside the original image.

