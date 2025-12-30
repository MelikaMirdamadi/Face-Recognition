# Face Recognition with FAISS

## Project Goal
This project implements a face recognition system that uses FAISS (Facebook AI Similarity Search) for efficient similarity search. The goal is to process a dataset of face images, extract facial embeddings (numerical representations of faces), store them in a vector database, and then identify unknown faces by finding the most similar known face in the database. Upon finding a match, the system returns the ID (name) of the recognized person along with a similarity score.

## Key Features
- **Face Detection and Embedding**: Uses InsightFace library with the Buffalo_L model to detect faces and extract 512-dimensional embeddings.
- **Vector Search**: Employs FAISS for fast similarity search using cosine similarity.
- **Dataset Management**: Organizes face images in folders named after individuals.
- **Scalable**: Can handle large datasets efficiently once indexed.

## Project Roadmap

*****
The current setup uses the buffalo_l model from insightface for both face detection and embedding extraction. Internally, this model employs RetinaFace for the detection step (to locate faces in images) and ArcFace for generating the 512-dimensional embeddings used in recognition.
*****


```
graph TD;
    A[Project Initialization] --> B[Environment Setup];
    B --> C[Prepare Dataset];
    C --> D[Download InsightFace Model];
    D --> E[Build Face Vector Database];
    E --> F[Perform Face Recognition];
    F --> G[Run Demo/App];
    G --> H[Deployment/Usage];
    H --> I[End];
    
    B --> J[Install Python 3.8+];
    B --> K[Create Virtual Environment];
    K --> L[Activate Venv];
    L --> M[Install Dependencies from requirements.txt];
    
    C --> N[Create dataset/ folder];
    N --> O[Organize images in subfolders by person];
    O --> P[Ensure one clear face per image];
    
    D --> Q[Buffalo_L model downloads automatically];
    Q --> R[Manual download if needed];
    
    E --> S[Run main.py to build FAISS index];
    S --> T[Extract embeddings from dataset];
    T --> U[Store in database/faiss.index];
    U --> V[Save labels in database/labels.txt];
    
    F --> W[Use FaceVectorDB.search() method];
    W --> X[Input query image];
    X --> Y[Return person name and similarity score];
    
    G --> Z[streamlit run app.py];
    Z --> AA[Upload images for recognition];
    AA --> BB[Browse dataset grid];
```

## Project Steps

### 1. Environment Setup
- **Install Python**: Ensure Python 3.8+ is installed.
- **Create Virtual Environment**:
  ```
  python -m venv venv
  venv\Scripts\activate  # On Windows
  ```
- **Install Dependencies**:
  ```
  pip install -r requirements.txt
  ```
  Key packages: `faiss-cpu`, `insightface`, `opencv-python`, `numpy`, `onnxruntime`.

### 2. Prepare Dataset
- Create a `dataset/` folder in the project root.
- Inside `dataset/`, create subfolders named after each person (e.g., `Aaron_Peirsol/`, `Abdoulaye_Wade/`).
- Place face images (JPG, PNG) in each person's folder. Each image should contain one clear face.

### 3. Download InsightFace Model (if needed)
- The Buffalo_L model downloads automatically on first run.
- If network issues occur, download manually from: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
- Extract to: `C:\Users\<username>\.insightface\models\buffalo_l`

### 4. Build the Face Vector Database
- Run the main script to process the dataset and build the FAISS index:
  ```
  python main.py
  ```
- This will:
  - Extract embeddings from all images in `dataset/`.
  - Store embeddings in `database/faiss.index`.
  - Save person labels in `database/labels.txt`.

### 5. Perform Face Recognition
- To search for a face in a new image:
  - Modify `main.py` or create a new script to call `db.search('path/to/query_image.jpg')`.
  - The method returns a list of tuples: `[(person_name, similarity_score), ...]`
  - The highest similarity score indicates the best match.

### 6. Usage Example
```python
from vectordb import FaceVectorDB

# Initialize database
db = FaceVectorDB()

# Build index (if not already done)
if len(db.labels) == 0:
    db.build_index()

# Search for a face
results = db.search('path/to/new_face.jpg')
if results:
    print(f"Recognized: {results[0][0]} with similarity {results[0][1]}")
else:
    print("No face found or no match.")
```

## File Structure
```
Face-Recognition/
├── main.py                 # Main script to build index and test
├── vectordb.py             # FaceVectorDB class for embedding and search
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── dataset/                # Face images organized by person
│   ├── Person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── Person2/
│       └── image3.jpg
└── database/               # FAISS index and labels
    ├── faiss.index
    └── labels.txt
```

## Notes
- Ensure images have clear, front-facing faces for best results.
- The system assumes one face per image; multiple faces may require modifications.
- For production, consider adding confidence thresholds and handling edge cases.
- FAISS index is saved to disk, so rebuilding is only needed when adding new faces.

## Demo
To run a visual demo for presentation:
```
streamlit run app.py
```
This launches a web interface where you can:
- Upload images for face recognition and see the matching person with similarity score.
- View all dataset images in a grid layout for easy browsing.

### Demo Features
- **Face Recognition**: Upload a query image and get instant recognition results.
- **Dataset Grid**: Browse all stored faces organized by person.
- **Similarity Scores**: View confidence levels for matches.

Make sure `app.py` is present in the project root. If not, create a simple Streamlit app using the `FaceVectorDB` class.