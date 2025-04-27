import chromadb
from pathlib import Path
import sys
import traceback

# --- Configuration (MUST match database-maintain.py/web_app.py) ---
VECTOR_DB_PATH_STR = "F:/Code/~code/vector_db"
# --- End Configuration ---

print(f"Attempting to connect to ChromaDB at: {VECTOR_DB_PATH_STR}")
db_path = Path(VECTOR_DB_PATH_STR)

if not db_path.exists() or not db_path.is_dir():
    print(f"ERROR: Vector DB directory not found at '{VECTOR_DB_PATH_STR}'.")
    print("Ensure the path is correct and the database has been initialized.")
    sys.exit(1)

try:
    # Connect to the persistent client
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH_STR)
    print(f"Successfully connected to client.")

    # List all collections
    print("\n--- Listing Collections ---")
    list_of_collections = client.list_collections()

    if not list_of_collections:
        print("No collections found in this ChromaDB instance.")
    else:
        print(f"Found {len(list_of_collections)} collection(s):")
        for i, collection_obj in enumerate(list_of_collections):
            # The objects in the list have attributes like 'name'
            print(f"  {i+1}. Name: '{collection_obj.name}'")
            # You can print other attributes if needed, e.g., collection_obj.metadata
            # print(f"     Metadata: {collection_obj.metadata}")

    print("--- End Listing ---")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    traceback.print_exc()

