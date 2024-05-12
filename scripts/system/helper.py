from __init__ import *

def load_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    return image

def preprocess_image(image):
    if image is None: return None
    transform_step = transforms.Compose([
        transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_step(image).unsqueeze(0)

def add_commas(array):
    # Convert the numpy array to a list of strings
    string_list = [str(element) for element in array]

    # Join the elements with commas
    result = ','.join(string_list)

    return '[' + result + ']'