from vectordb import FaceVectorDB
import sys

def main():
    if len(sys.argv) > 1:
        query_image = sys.argv[1]
    else:
        query_image = "./test.jpg"

    db = FaceVectorDB()
    # Build the index if not exists
    if db.db_type == 'faiss':
        if len(db.labels) == 0:
            print("Building FAISS index...")
            db.build_index()
            print("Index built.")
        else:
            print("FAISS index already exists.")
    elif db.db_type == 'qdrant':
        if db.client.count(db.collection_name).count == 0:
            print("Building Qdrant collection...")
            db.build_index()
            print("Collection built.")
        else:
            print("Qdrant collection already exists.")

    # Search for the query image
    if query_image:
        results = db.search(query_image)
        if results:
            print(f"Recognized: {results[0][0]} with similarity {results[0][1]:.4f}")
        else:
            print("No face found or no match.")
    else:
        print("No query image provided.")

if __name__ == "__main__":
    main()