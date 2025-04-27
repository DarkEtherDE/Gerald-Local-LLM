"""
Indexes text-based files recursively by chunking content, generating embeddings,
and storing them in a ChromaDB vector database for semantic search.
Handles updates based on file modification times.
"""

import sys
from pathlib import Path
import time
import argparse
from sentence_transformers import SentenceTransformer
import logging # Use logging for better output control
import chromadb
# --- End NEW Imports ---

# --- Configuration ---

# == REQUIRED: Embedding Configuration ==
# Choose the SAME embedding model to be used by the query script
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Example SentenceTransformer model
# EMBEDDING_MODEL_NAME = 'nomic-embed-text' # Example if using Ollama for embeddings

# == REQUIRED: Vector Database Configuration ==
# Determine paths relative to *this* script's location
try:
    script_path = Path(__file__).resolve()
except NameError:
    script_path = Path(sys.argv[0]).resolve()
script_dir = script_path.parent

VECTOR_DB_PATH = script_dir / "vector_db" # Directory where ChromaDB stores data
COLLECTION_NAME = "file_chunks" # Collection name

# == Indexing Configuration ==
# Add or remove extensions you want to index
RELEVANT_EXTENSIONS = {
    ".txt", ".py", ".md", ".csv", ".json",
    ".html", ".css", ".js", ".yaml", ".yml",
    ".rst", ".sh", ".cfg", ".ini", ".log",
    ".eml", ".mbox",
}
# Chunking parameters (adjust as needed)
CHUNK_SIZE = 500 # Approximate characters per chunk
CHUNK_OVERLAP = 50 # Approximate characters overlap between chunks

# == Logging Configuration ==
LOG_FILE_NAME = "indexing_semantic_log.txt"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- End Configuration ---

# --- Logging Setup ---
def setup_logging(log_dir):
    """Sets up logging to file and console."""
    log_file = log_dir / LOG_FILE_NAME
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Also print logs to console
        ]
    )
    logging.info("--- Semantic Indexing session started ---")
    return log_file

# --- Helper function for Chunking ---
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into chunks using basic textwrap."""
    # Consider replacing with more sophisticated methods if needed
    # e.g., splitting by paragraphs first, then wrapping.
    from textwrap import wrap
    if not text: # Handle empty content
        return []
    # Basic wrap - might split mid-sentence
    chunks = wrap(text, width=chunk_size, subsequent_indent=" " * overlap, break_long_words=False, break_on_hyphens=False)
    # Filter out potentially empty chunks resulting from wrapping
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# --- Helper function for Ollama Embeddings (Optional) ---
# def get_ollama_embedding(text_chunk, model_name=EMBEDDING_MODEL_NAME):
#     import requests
#     try:
#         response = requests.post("http://localhost:11434/api/embeddings", json={
#             "model": model_name,
#             "prompt": text_chunk
#         }, timeout=60) # Add timeout
#         response.raise_for_status()
#         embedding = response.json().get("embedding")
#         if embedding:
#             return embedding
#         else:
#             logging.error(f"Ollama embedding response missing 'embedding' key: {response.json()}")
#             return None
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error connecting to Ollama embedding endpoint: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Error getting Ollama embedding: {e}")
#         return None

# --- Main Indexing Function ---
def index_directory(source_dir, vector_db_client, embedding_model_instance, script_path):
    # ... (existing setup: get collection, get existing metadata) ...
    try:
        logging.info(f"Getting/Creating ChromaDB collection: {COLLECTION_NAME}")
        collection = vector_db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            # Optional: Add metadata to the collection itself if desired
            # metadata={"hnsw:space": "cosine"} # Example if using cosine distance
        )
        logging.info(f"Collection '{COLLECTION_NAME}' obtained.")
    except Exception as e:
        logging.error(f"Failed to get/create collection '{COLLECTION_NAME}': {e}", exc_info=True)
        return 0, 0, 0 # Indicate failure

    # --- Get existing metadata (as before) ---
    try:
        existing_metadata = collection.get(include=['metadatas'])
        # Create a lookup dictionary: filename -> last_modified timestamp
        last_indexed_times = {
            meta['filename']: meta.get('last_modified', 0)
            for meta in existing_metadata.get('metadatas', []) if meta and 'filename' in meta
        }
        logging.info(f"Found metadata for {len(last_indexed_times)} previously indexed files.")
    except Exception as e:
        # Handle case where collection might exist but is empty or GET fails
        logging.warning(f"Could not retrieve existing metadata or collection is empty: {e}. Will index all found files.")
        last_indexed_times = {}


    # --- File Processing Loop ---
    source_dir_path = Path(source_dir)
    files_processed = 0
    files_skipped = 0
    files_failed = 0
    script_dir = script_path.parent
    vector_db_dir_abs = (script_dir / VECTOR_DB_PATH).resolve() # Use absolute path for comparison

    logging.info(f"Starting recursive scan of directory: {source_dir_path}")
    for item in source_dir_path.rglob('*'):
        # --- Skip directories and unwanted paths (as before) ---
        if not item.is_file():
            continue
        # Resolve item path to compare against absolute DB path
        item_abs = item.resolve()
        if item_abs.is_relative_to(vector_db_dir_abs):
             logging.debug(f"Skipping file inside vector DB directory: {item}")
             continue
        if item_abs == script_path:
             logging.debug(f"Skipping the script file itself: {item}")
             continue
        if item.suffix.lower() not in RELEVANT_EXTENSIONS:
             logging.debug(f"Skipping file with non-relevant extension: {item}")
             continue

        # --- Process File (Modified Part) ---
        try:
            relative_filename = item.relative_to(source_dir_path)
            mtime = item.stat().st_mtime

            # Check if file needs update
            last_indexed_time = last_indexed_times.get(str(relative_filename), 0)
            if mtime <= last_indexed_time:
                logging.debug(f"Skipping unchanged file: {relative_filename}")
                files_skipped += 1
                continue

            logging.info(f"Processing '{relative_filename}'...")
            files_processed += 1

            # Read file content (as before)
            try:
                file_content_str = item.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 decode failed for {relative_filename}, trying latin-1.")
                file_content_str = item.read_text(encoding='latin-1')
            except Exception as read_err:
                 logging.error(f"Failed to read file {relative_filename}: {read_err}")
                 files_failed += 1
                 continue # Skip to next file

            # Chunk text (as before)
            chunks = textwrap.wrap(file_content_str, width=CHUNK_SIZE, replace_whitespace=False, drop_whitespace=False)
            if not chunks:
                 logging.warning(f"File {relative_filename} resulted in zero chunks.")
                 continue # Skip if no content

            # Generate embeddings (as before)
            logging.debug(f"Generating {len(chunks)} embeddings for {relative_filename}...")
            embeddings = embedding_model_instance.encode(chunks, show_progress_bar=False)

            # --- START NEW: Determine Subdirectory ---
            relative_path_parts = relative_filename.parts
            if len(relative_path_parts) > 1: # It's in at least one subdirectory
                subdirectory = relative_path_parts[0]
            else: # It's directly in the root source directory
                subdirectory = "root" # Use "root" for files directly in the dataset base dir
            logging.debug(f"Assigning subdirectory '{subdirectory}' for {relative_filename}")
            # --- END NEW: Determine Subdirectory ---

            # Prepare data for ChromaDB (Modified)
            ids = [f"{str(relative_filename)}_{i}" for i in range(len(chunks))]
            metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "filename": str(relative_filename),
                    "chunk_index": i,
                    "last_modified": mtime,
                    "subdirectory": subdirectory # <<< ADDED METADATA FIELD
                }
                metadatas.append(chunk_metadata)

            # Delete old entries for this file (as before)
            try:
                 collection.delete(where={"filename": str(relative_filename)})
                 logging.debug(f"Deleted old entries for {relative_filename}")
            except Exception as del_err:
                 # Log error but proceed, maybe old entries didn't exist
                 logging.warning(f"Could not delete old entries for {relative_filename} (may not exist): {del_err}")

            # Add new data (as before)
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            logging.debug(f"Added {len(chunks)} chunks for {relative_filename}")

        except Exception as e:
            logging.error(f"Failed to process file {item}: {e}", exc_info=True)
            files_failed += 1

    # --- End Loop ---
    logging.info("Finished directory scan.")
    return files_processed, files_skipped, files_failed
# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Recursively index text files into a ChromaDB vector database using embeddings.",
        epilog=f"Example: python {script_path.name} /path/to/your/dataset"
    )
    # --- NEW: Argument for source directory ---
    parser.add_argument(
        "source_directory",
        type=str,
        help="The path to the directory containing files to index."
    )
    # Optional: Add arguments for vector DB path, collection name, embedding model if needed

    args = parser.parse_args()

    source_dir = Path(args.source_directory).resolve()
    if not source_dir.is_dir():
        print(f"Error: Source directory not found or is not a directory: {source_dir}")
        sys.exit(1)

    setup_logging(script_dir) # Setup logging using script's directory

    # --- Initialize ChromaDB Client and Embedding Model ---
    try:
        logging.info(f"Initializing ChromaDB client from path: {VECTOR_DB_PATH}")
        # Ensure the parent directory exists if needed by PersistentClient
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        vector_db_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        # Load the SentenceTransformer model (or configure Ollama access here)
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded successfully.")

    except Exception as e:
        logging.error(f"Fatal error during initialization: {e}", exc_info=True)
        print(f"Error during initialization, check log file: {LOG_FILE_NAME}. Exiting.")
        sys.exit(1)

    # --- Run Indexing ---
    try:
        index_directory(source_dir, vector_db_client, embedding_model_instance, script_path)
    except KeyboardInterrupt:
        logging.warning("Indexing interrupted by user.")
        print("\nIndexing interrupted.")
    except Exception as e:
        logging.error("An unexpected error occurred during indexing.", exc_info=True)
        print(f"An unexpected error occurred, check log file: {LOG_FILE_NAME}")
    finally:
        # Optional: Cleanup resources if necessary
        # ChromaDB PersistentClient might not need explicit closing
        logging.info("Indexing process finished.")


if __name__ == "__main__":
    main()
