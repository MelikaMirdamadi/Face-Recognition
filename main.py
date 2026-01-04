from vectordb import FaceVectorDB
import sys

def main():
    if len(sys.argv) > 1:
        query_image = sys.argv[1]
    else:
        query_image = "./test.jpg"

    db = FaceVectorDB()
    
    # Always rebuild the index to include any new images in the dataset
    print("Rebuilding index/collection...")
    db.build_index()
    print("Index/collection rebuilt.")
    
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