from recommendation_pipeline import *
from helper import *
import streamlit as st

def main():
    col_width = 200
    img_display_size = col_width
    num_cols = 5
    st.title("Outfit Recommendation App Using pgvector")
    upload = st.file_uploader("Choose an item to upload")
    image = None
    if upload is not None:
        image = Image.open(upload)
        st.subheader("Here are recommended combinations might be suitable for you")
    if image is None:
        st.subheader("Please upload an item to get recommendations")
    else:
        pipeline = RecommendationPipeline()
        recommended_outfits = pipeline.recommend(image)
        for outfit in recommended_outfits:
            columns = st.columns([1, 1, 1, 1, 1])
            with columns[0]:
                st.image(image, width=col_width)
            for i, image_path in enumerate(outfit):
                with columns[(i + 1) % num_cols]:
                    img = Image.open(os.path.join("../../itemsdb", image_path))
                    img = img.resize((img_display_size, img_display_size))
                    st.image(img, width=col_width)
            
    

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main()