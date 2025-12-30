import os
import numpy as np
import faiss
import insightface
from insightface.app import FaceAnalysis
import cv2

class FaceVectorDB:
    def __init__(self, index_path='database/faiss.index', dataset_path='dataset', threshold=0.5):
        self.index_path = index_path
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.app = FaceAnalysis(name='buffalo_l')  # Use buffalo_l model
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # Prepare for CPU

        # Load or create FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Assuming 512-dimensional embeddings from insightface
            self.index = faiss.IndexFlatIP(512)  # Inner product for cosine similarity

        self.labels = []
        self.load_labels()

    def extract_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        # Take the first face
        embedding = faces[0].embedding.astype(np.float32)
        # Normalize the embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def build_index(self):
        self.index = faiss.IndexFlatIP(512)
        self.labels = []
        for person_folder in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_folder)
            if os.path.isdir(person_path):
                print(f"Processing folder: {person_folder}")
                for image_file in os.listdir(person_path):
                    if image_file.endswith(('.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(person_path, image_file)
                        print(f"Processing image: {image_path}")
                        embedding = self.extract_embedding(image_path)
                        if embedding is not None:
                            self.index.add(np.array([embedding]))
                            self.labels.append(person_folder)
                            print(f"Added embedding for {person_folder}")
                        else:
                            print(f"No face found in {image_path}")
        print(f"Total embeddings added: {len(self.labels)}")
        # Save index
        faiss.write_index(self.index, self.index_path)
        # Save labels (simple way, save to a file)
        with open('database/labels.txt', 'w') as f:
            for label in self.labels:
                f.write(label + '\n')

    def load_labels(self):
        if os.path.exists('database/labels.txt'):
            with open('database/labels.txt', 'r') as f:
                self.labels = [line.strip() for line in f]

    def search(self, query_image_path, k=1):
        embedding = self.extract_embedding(query_image_path)
        if embedding is None:
            return None
        distances, indices = self.index.search(np.array([embedding]), k)
        results = []
        for i in range(k):
            if indices[0][i] != -1:
                results.append((self.labels[indices[0][i]], distances[0][i]))
        
        # Apply threshold: if top result below threshold, mark as unknown
        if results and results[0][1] < self.threshold:
            return [("unknown", results[0][1])]
        return results

# Usage example
if __name__ == "__main__":
    db = FaceVectorDB()
    # If index doesn't exist, build it
    if not os.path.exists(db.index_path):
        db.build_index()
    else:
        db.load_labels()
    # Search for a query image
    # query_results = db.search('path/to/query/image.jpg')
    # print(query_results)