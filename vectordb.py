import os
import numpy as np
import faiss
import insightface
from insightface.app import FaceAnalysis
import cv2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class FaceVectorDB:
    
    # to use faiss database
    # db_type='faiss' db_path='database/faiss.index'

    def __init__(self, db_type='qdrant', db_path='face_collection', dataset_path='dataset', threshold=0.5, qdrant_url='http://localhost:6333'):
        self.db_type = db_type
        self.db_path = db_path
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.app = FaceAnalysis(name='buffalo_l')  # Use buffalo_l model
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # Prepare for CPU

        if self.db_type == 'faiss':
            # Load or create FAISS index
            if os.path.exists(self.db_path):
                self.index = faiss.read_index(self.db_path)
            else:
                # Assuming 512-dimensional embeddings from insightface
                self.index = faiss.IndexFlatIP(512)  # Inner product for cosine similarity

            self.labels = []
            self.load_labels()
        elif self.db_type == 'qdrant':
            self.client = QdrantClient(url=qdrant_url)
            self.collection_name = db_path  # db_path is collection name for qdrant
            # Create collection if not exists
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
        else:
            raise ValueError("db_type must be 'faiss' or 'qdrant'")

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
        if self.db_type == 'faiss':
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
            faiss.write_index(self.index, self.db_path)
            # Save labels (simple way, save to a file)
            with open('database/labels.txt', 'w') as f:
                for label in self.labels:
                    f.write(label + '\n')
        elif self.db_type == 'qdrant':
            points = []
            id_counter = 0
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
                                points.append(PointStruct(
                                    id=id_counter,
                                    vector=embedding.tolist(),
                                    payload={"label": person_folder}
                                ))
                                id_counter += 1
                                print(f"Added embedding for {person_folder}")
                            else:
                                print(f"No face found in {image_path}")
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Total embeddings added: {len(points)}")

    def load_labels(self):
        if self.db_type == 'faiss':
            if os.path.exists('database/labels.txt'):
                with open('database/labels.txt', 'r') as f:
                    self.labels = [line.strip() for line in f]

    def search(self, query_image_path, k=1):
        embedding = self.extract_embedding(query_image_path)
        if embedding is None:
            return None
        
        if self.db_type == 'faiss':
            distances, indices = self.index.search(np.array([embedding]), k)
            results = []
            for i in range(k):
                if indices[0][i] != -1:
                    results.append((self.labels[indices[0][i]], distances[0][i]))
        elif self.db_type == 'qdrant':
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=k
            )
            results = [(hit.payload['label'], hit.score) for hit in search_result]
        
        # Apply threshold: if top result below threshold, mark as unknown
        if results and results[0][1] < self.threshold:
            return [("unknown", results[0][1])]
        return results

# Usage example
if __name__ == "__main__":
    # For FAISS
    db_faiss = FaceVectorDB(db_type='faiss')
    if not os.path.exists(db_faiss.db_path):
        db_faiss.build_index()
    else:
        db_faiss.load_labels()
    
    # For Qdrant
    db_qdrant = FaceVectorDB(db_type='qdrant', db_path='face_collection')
    if db_qdrant.client.count(db_qdrant.collection_name).count == 0:
        db_qdrant.build_index()
    
    # Search for a query image
    # query_results = db.search('path/to/query/image.jpg')
    # print(query_results)