import pandas as pd
from get_items import *

def preprocess_raw_data(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep="\t")
    # Standardize the label
    df['label'] = df['label'].apply(standardize_label)
    drop_cols = ['bounding_x', 'bounding_y', 'bounding_width', 'bounding_height']
    for col in df.columns:
        if col in drop_cols:
            df = df.drop(col, axis=1)
    return df

def preprocess_triplet_data(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep="\t")
    # Remove columns containing 'bounding_'
    filtered_df = df.drop(df.filter(regex='bounding_').columns, axis=1)
    # Standardize the label
    label_cols = df.filter(regex='label_').columns
    for col in label_cols:
        filtered_df[col] = filtered_df[col].apply(standardize_label)
    return filtered_df

if __name__ == '__main__':
    triplet_path_1 = "../datasets/triplet_train_p1.tsv"
    triplet_path_2 = "../datasets/triplet_train_p2.tsv"
    
    df_1 = preprocess_triplet_data(triplet_path_1)
    df_2 = preprocess_triplet_data(triplet_path_2)
    
    df = pd.concat([df_1, df_2])
    df = df.reset_index(drop=True)
    df.to_csv("../datasets/triplet_train.csv", index=False)