import streamlit as st
import os
from PIL import Image
from vectordb import FaceVectorDB

# Initialize the database
@st.cache_resource
def load_db():
    return FaceVectorDB()

db = load_db()

st.title("Face Recognition Demo")

# Section 1: Face Recognition Search
st.header("Face Recognition Search")
uploaded_file = st.file_uploader("Upload an image to recognize", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_query.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform search
    results = db.search("temp_query.jpg")
    if results:
        person, score = results[0]
        if person == "unknown":
            st.error(f"Unknown person detected (similarity: {score:.4f})")
        else:
            st.success(f"Recognized: **{person}** with similarity **{score:.4f}**")
        # Display the query image
        st.image(Image.open("temp_query.jpg"), caption="Query Image", width=200)
    else:
        st.error("No face found or no match in the database.")
        st.image(Image.open("temp_query.jpg"), caption="Query Image", width=200)

    # Clean up
    os.remove("temp_query.jpg")

# Section 2: Dataset Images Grid
st.header("Dataset Images")

dataset_path = "dataset"
if os.path.exists(dataset_path):
    # Collect all images
    images_data = []
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(person_path, image_file)
                    images_data.append((person_folder, image_path))

    if images_data:
        # Display in a grid
        cols = st.columns(4)  # 4 columns
        for i, (person, img_path) in enumerate(images_data):
            with cols[i % 4]:
                st.image(Image.open(img_path), caption=person, width=150)
    else:
        st.write("No images found in the dataset.")
else:
    st.write("Dataset folder not found.")

# Footer
st.write("---")
st.write("Built with Streamlit, InsightFace, and FAISS.")