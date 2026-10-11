import os
import glob
import random
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import urllib.request
import json



# pass the ae image to through resnet and gather confidence score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
weight = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weight).to(device)
model.eval()

preprocess = weight.transforms()

def resnet_confidence(csv_path):
    data_frame  = pd.read_csv(csv_path)
    image_path  = data_frame.iloc[1]['ae_image_path']
    image    = Image.open(image_path).convert('RGB')

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)

    # 6. Calculate Confidence
    # Squeeze the 1000 raw logits into a 0-100% probability distribution
    probabilities = F.softmax(logits, dim=1)
    
    # Extract the highest probability and its corresponding index
    confidence, class_idx = torch.max(probabilities, 1)

    # 7. Decode the result into human-readable text
    # The weights object secretly contains the dictionary of all 1000 ImageNet categories
    class_name = weight.meta["categories"][class_idx.item()]
    confidence_score = confidence.item() * 100

    print("=====================================")
    print(f"File: {image_path}")
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence_score:.2f}%")
    print("=====================================")


if __name__ == "__main__":
    # Replace with a real path to a .jpg on your machine
    sample_image = r"D:\Recent Advances\lens_test.csv" 
    resnet_confidence(sample_image)