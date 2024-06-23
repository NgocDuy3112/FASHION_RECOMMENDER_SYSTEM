import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import re
from argparse import ArgumentParser

def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)

def crop_image_from_url(image_url, bounding_x, bounding_y, bounding_width, bounding_height):
    try:
        # Send an HTTP request to the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Open the image using Pillow
        original_image = Image.open(BytesIO(response.content))
        
        def get_bounding_box(bounding_x, bounding_y, bounding_width, bounding_height):
            image_w, image_h = original_image.size
            left = bounding_x * image_w
            top = bounding_y * image_h
            right = (bounding_x + bounding_width) * image_w
            bottom = (bounding_y + bounding_height) * image_h
            return (left, top, right, bottom)

        # Extract the bounding box coordinates
        left, top, right, bottom = get_bounding_box(bounding_x, bounding_y, bounding_width, bounding_height)

        # Crop the image using the bounding box
        cropped_image = original_image.crop((left, top, right, bottom))

        # Return the cropped image
        return cropped_image

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def create_save_path(image_url, output_dir, label, train_or_test_string):
    save_path = os.path.join(output_dir, train_or_test_string, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(image_url))
    return save_path

def standardize_label(label):
    # Return a string with no space, lowercase, and replace "&" with "_" and only has 1 "_"
    new_label = re.sub(r'\s+', '_', label.lower().replace('&', '_'))
    new_label = re.sub(r'_+', '_', new_label)
    return new_label

def convert_to_float(value):
    if isinstance(value, float): return value
    return float(value.replace(',', '.'))

def main(args):
    tsv_file_path = args.tsv_file_path
    output_dir = args.output_dir
    df = pd.read_csv(tsv_file_path, sep='\t')
    train_or_test_string = re.split(r'[_\.]', tsv_file_path)[-2]
    cannot_save_paths = set()
    for _, row in df.iterrows():
        image_url = convert_to_url(row['image_signature'])
        cropped_image = crop_image_from_url(
            image_url, 
            float(convert_to_float(row['bounding_x'])), 
            float(convert_to_float(row['bounding_y'])), 
            float(convert_to_float(row['bounding_width'])), 
            float(convert_to_float(row['bounding_height']))
        )
        label = standardize_label(row['label'])
        save_path = create_save_path(image_url, output_dir, label, train_or_test_string)
        # Check if save_path exists
        if os.path.exists(save_path):
            print(f"An image with the following path: {save_path} already exists!")
            continue
        if cropped_image is not None:
            cropped_image.save(save_path)
            print(f"An image with the following path: {save_path} has been successfully saved!")
        else:
            cannot_save_paths.add(save_path)
            print(f"An image with the following path: {save_path} could not be saved!")
            continue
    if len(cannot_save_paths) > 0:
        print(f"The following images could not be saved: ")
        for path in cannot_save_paths:
            print(path)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tsv_file_path', type=str, help='path to tsv file')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    args = parser.parse_args()
    main(args)