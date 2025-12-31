import streamlit as st
from vectordb import FaceVectorDB
import os


# Initialize the database
@st.cache_resource
def load_db():
    db = FaceVectorDB()
    return db

db = load_db()

st.title("Face Recognition Demo")
st.write("Upload an image to recognize the person using FAISS similarity search.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_query.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Perform search
    results = db.search(temp_path, k=3)  # Get top 3 matches

    if results:
        st.success("Recognition Results:")
        for i, (name, similarity) in enumerate(results, 1):
            st.write(f"{i}. **{name}** - Similarity: {similarity:.4f}")
    else:
        st.error("No face found or no match in the database.")

    # Clean up temp file
    os.remove(temp_path)

st.write("---")
st.write("**Database Info:**")
st.write(f"- Total faces in database: {len(db.labels)}")
st.write("- Persons:", ", ".join(set(db.labels)))

st.write("**How it works:**")
st.write("1. Face detection and embedding extraction using InsightFace.")
st.write("2. Similarity search in FAISS vector database.")
st.write("3. Returns the most similar person's ID and score.")