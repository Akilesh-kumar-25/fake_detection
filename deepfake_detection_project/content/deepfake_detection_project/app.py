
import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
# Assuming the model file is in the 'model' subfolder
model_path = "model/deepfake_model.pth"

# Load the same model architecture used for training
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

# Load the state dict, handling potential missing keys if architecture differs slightly
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except RuntimeError as e:
    print(f"Error loading model state_dict: {e}")
    print("Attempting to load with strict=False...")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)


model.to(device)
model.eval() # Set the model to evaluation mode

# Define the same transforms used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_image(img: Image.Image) -> str:
    """
    Classifies an uploaded image as real, AI-generated, or drawn and returns a descriptive paragraph.

    Args:
        img: The uploaded image as a PIL Image object.

    Returns:
        A string containing a short paragraph describing the classification.
    """
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    # Assuming the order is the same as during training
    # It's best to get the class labels from the dataset if possible,
    # but for this example, we'll use the assumed order.
    class_labels = ['ai', 'drawn', 'real']
    predicted_class = class_labels[pred]

    # Generate the descriptive paragraph based on the prediction
    if predicted_class == 'real':
        return "Based on the analysis, this image appears to be a real photograph captured by a camera."
    elif predicted_class == 'ai':
        return "Based on the analysis, this image is likely AI-generated. It exhibits characteristics often found in images created by artificial intelligence models."
    elif predicted_class == 'drawn':
        return "Based on the analysis, this image seems to be hand-drawn or created using drawing tools. It displays traits typical of artistic creations."
    else:
        return "Unable to classify the image."

# Example usage (optional, could be removed for the final app.py)
# if __name__ == "__main__":
#     # Example of how to use the function with a local image file
#     # Make sure to have a test image available in your project structure
#     test_image_path = "path/to/your/test_image.jpg" # Replace with a valid path
#     if os.path.exists(test_image_path):
#         prediction = classify_image(Image.open(test_image_path))
#         print(f"Prediction for {test_image_path}: {prediction}")
#     else:
#         print(f"Test image not found at {test_image_path}")


interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🔍 DeepFake Detection",
    description="Upload an image to detect if it's REAL, AI-generated, or HAND-DRAWN and get a descriptive analysis.",
    # Examples would need to be paths relative to where the app is run,
    # or you would need to include sample images in your project structure.
    # For simplicity in deployment, examples might be omitted or handled differently.
    # examples=[
    #     ["path/to/sample_real.jpg"],
    #     ["path/to/sample_ai.jpg"],
    #     ["path/to/sample_drawn.jpg"]
    # ]
)

if __name__ == "__main__":
    interface.launch()

