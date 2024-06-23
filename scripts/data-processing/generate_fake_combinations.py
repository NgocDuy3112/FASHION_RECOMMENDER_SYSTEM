from PIL import Image
import pandas as pd
import os
import random
import threading

random.seed(42)

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

def process_image(image_path, df, concatenated_images_info):
    try:
        with Image.open(image_path) as image:
            return image.copy()
    except Exception as e:
        print(f"Error opening image '{image_path}': {e}")
        return None

def process_group(image_signature, group, df, concatenated_images_info):
    images = []
    image_paths = group['image_path'].tolist()
    try:
        for image_path in image_paths:
            image = process_image(image_path, df, concatenated_images_info)
            if image:
                images.append(image)
    except Exception as e:
        print(f"Error processing group for signature '{image_signature}': {e}")
        return

    if images:
        random_index = random.randint(0, len(images) - 1)
        selected_label = df.loc[df['image_path'] == group.iloc[random_index]['image_path'], 'label'].iloc[0]
        same_label_images = [img for img, path in zip(images, image_paths) if
                             df.loc[df['image_path'] == path, 'label'].iloc[0] == selected_label]

        new_image = random.choice(same_label_images)
        images[random_index] = new_image

        concatenated_image = concatenate_images(images)

        output_path = os.path.join("../../complete-the-look-dataset/outfit-classification/test-restricted/0", 
                                   f"{image_signature}_fake.jpg")
        try:
            concatenated_image.save(output_path)
            concatenated_images_info.append({'image_signature': image_signature,
                                             'concatenated_image_path': output_path})
            print(f"Combination {image_signature} fake saved!")
        except Exception as e:
            print(f"Error saving concatenated image for signature '{image_signature}': {e}")
            
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
    return df

if __name__ == "__main__":
    df_path = "../../complete-the-look-dataset/datasets/preprocessed/test.csv"
    parent_path = "../../complete-the-look-dataset/items/test/"
    df = process_dataframe(df_path, parent_path)
    
    concatenated_images_info = []

    threads = []
    for image_signature, group in df.groupby('image_signature'):
        t = threading.Thread(target=process_group, args=(image_signature, group, df, concatenated_images_info))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    concatenated_images_df = pd.DataFrame(concatenated_images_info)
