from vectordb import FaceVectorDB

def main():
    db = FaceVectorDB()
    # Build the index if not exists or if labels are empty
    if len(db.labels) == 0:
        print("Building index...")
        db.build_index()
        print("Index built.")
    else:
        print("Index already exists.")

    # Example search
    # Assuming you have a query image
    results = db.search('path/to/query.jpg')
    if results:
        print(f"Recognized: {results[0][0]} with similarity {results[0][1]}")
    else:
        print("No face found or no match.")

if __name__ == "__main__":
    main()