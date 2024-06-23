import pandas as pd
import requests
import os
import re
from argparse import ArgumentParser
import concurrent.futures

def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)

def standardize_label(label):
    # Return a string with no space, lowercase, and replace "&" with "_" and only has 1 "_"
    new_label = re.sub(r'\s+', '_', label.lower().replace('&', '_'))
    new_label = re.sub(r'_+', '_', new_label)
    return new_label

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None
    
def create_save_path(image_url, output_dir, train_or_test_string):
    save_path = os.path.join(output_dir, train_or_test_string)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(image_url))
    return save_path

def main(args):
    tsv_file_path = args.tsv_file_path
    output_dir = args.output_dir
    df = pd.read_csv(tsv_file_path, sep='\t').drop_duplicates(subset='image_signature')
    train_or_test_string = os.path.splitext(os.path.basename(tsv_file_path))[0]
    cannot_save_paths = set()
    
    # Download images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _, row in df.iterrows():
            image_url = convert_to_url(row['image_signature'])
            save_path = create_save_path(image_url, output_dir, train_or_test_string)
            
            if not os.path.exists(save_path):
                futures.append(executor.submit(download_image, image_url, save_path))
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            save_path = future.result()
            if save_path is not None:
                print(f"Image saved: {save_path}")
            else:
                cannot_save_paths.add(save_path)
                print(f"Error saving image: {save_path}")
    
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
