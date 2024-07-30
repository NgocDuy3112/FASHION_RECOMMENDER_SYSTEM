from recommendation_pipeline import *
from helper import *
import streamlit as st

def main():
    col_width = 200
    img_display_size = col_width
    num_cols = 5
    st.title("Outfit Recommendation App")
    big_columns = st.columns([3, 1])
    with big_columns[0]:
        upload = st.file_uploader("Upload an item for recommendation")
    with big_columns[1]:
        n_items = st.slider("How many items do you want from a combination?", 3, 5, 5)
        k = st.slider("How many combinations do you want to generate?", 3, 10, 4)
        generate = st.button("Generate")
        
    image = None
    if upload is not None:
        image = Image.open(upload)
    if generate:
        if image is None:
            st.subheader("Please upload an item to get recommendations")
        else:
            st.subheader("Here are recommended combinations might be suitable for you")
            pipeline = RecommendationPipeline()
            recommended_outfits = pipeline.recommend("test", image, n_items, k)
            for outfit in recommended_outfits:
                columns = st.columns([1, 1, 1, 1, 1])
                with columns[0]:
                    st.image(image, width=col_width)
                # Debugging
                # for i, (image_path, category) in enumerate(outfit):
                #     with columns[(i + 1) % num_cols]:
                #         img = Image.open(os.path.join("../../itemsdb", image_path))
                #         img = img.resize((img_display_size, img_display_size))
                #         st.text(category)
                #         st.image(img, width=col_width)
                for i, item in enumerate(outfit):
                    with columns[(i + 1) % num_cols]:
                        img = Image.open(os.path.join("../../testdb", item[0]))
                        img = img.resize((img_display_size, img_display_size))
                        st.image(img, width=col_width)
    

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    main()
