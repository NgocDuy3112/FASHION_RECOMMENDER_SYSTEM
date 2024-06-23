from PIL import Image
import pandas as pd
import os

def concatenate_images(images):
    # Find the dimensions of the largest image
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Resize each image to the dimensions of the largest image
    resized_images = [image.resize((max_width, max_height)) for image in images]

    # Concatenate the resized images
    total_width = sum(image.width for image in resized_images)
    concatenated_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in resized_images:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return concatenated_image

def create_folder_structure(root_folder, subfolder):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
        
def is_a_full_combination(combination):
    if 'shirts_tops' in combination: return any(item in combination for item in ['pants', 'skirts', 'shorts'])
    if 'dresses' in combination:
        if any(item in combination for item in ['shirts_tops', 'pants', 'shorts', 'skirts']):
            return False
        else: return True
    return False
        
def process_dataframe(df_path, parent_path):
    df = pd.read_csv(df_path, engine='c')
    df['image_path'] = parent_path + df['label'] + "/" + df['image_signature'] + ".jpg"
    
    remove_items_list = ['belts', 'gloves_mittens', 'jumpsuits_rompers', 'neckties', 'rings', 'socks', 'stockings', 'swimwear']
    # List all image_signatures contains at least one of the values in remove_items_list
    removed_df = df[df['label'].isin(remove_items_list)]
    # Removed all image_signatures in df contains in removed_df
    df = df[~df['image_signature'].isin(removed_df['image_signature'])]
    # Remove combinations contains less than 3 items and more than 5 items
    grouped_data = df.groupby('image_signature')['label'].apply(list).reset_index()
    filtered_data = grouped_data[(grouped_data['label'].apply(len) >= 3) & (grouped_data['label'].apply(len) <= 5) & (grouped_data['label'].apply(is_a_full_combination))]
    df = df[df['image_signature'].isin(filtered_data['image_signature'])]
    # Remove all combinations are not full combination
    return df
        
if __name__ == "__main__":
    df_path = "../../complete-the-look-dataset/datasets/preprocessed/train.csv"
    parent_path = "../../complete-the-look-dataset/items/train/"
    df = process_dataframe(df_path, parent_path)
    
    concatenated_images_info = []
    
    train_combinations = df.groupby('image_signature')

    for image_signature, group in train_combinations:
        images = []
        for image_path in group['image_path']:
            try:
                image = Image.open(image_path)
                images.append(image)
            except Exception as e:
                print(f"Error opening image '{image_path}': {e}. Ignore combination: {image_signature}")
                images = []  # Reset images list if any image fails to open
                break  # Skip to the next combination

        if images:  # If there are valid images
            concatenated_image = concatenate_images(images)
            output_path = os.path.join("../../complete-the-look-dataset/outfit-classification/train-restricted/1", f"{image_signature}.jpg")
            try:
                concatenated_image.save(output_path)
                concatenated_images_info.append({'image_signature': image_signature, 'concatenated_image_path': output_path})
                print(f"Combination {image_signature} saved!")
            except Exception as e:
                print(f"Error saving concatenated image for signature '{image_signature}': {e}")

    concatenated_images_df = pd.DataFrame(concatenated_images_info)
    print("Saving completed!")