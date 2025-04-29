import sys
from pathlib import Path
import time
import argparse
import textwrap
from sentence_transformers import SentenceTransformer
import logging
import chromadb
import configparser
import json

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
#Vector Database Configuration ==
try: script_path = Path(__file__).resolve()
except NameError: script_path = Path(sys.argv[0]).resolve()
script_dir = script_path.parent
VECTOR_DB_PATH = script_dir / "vector_db"
DEFAULT_COLLECTION_NAME = "file_chunks"
# == Indexing Configuration ==
RELEVANT_EXTENSIONS = { ".txt", ".py", ".md", ".csv", ".json", ".html", ".css", ".js", ".yaml", ".yml", ".rst", ".sh", ".cfg", ".ini", ".log", ".eml", ".mbox", ".pdf", }
CHUNK_SIZE = 500; CHUNK_OVERLAP = 50
# Logging Configuration 
LOG_FILE_NAME = "semantic_log.txt"; LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
CONFIG_FILE_NAME = "config.ini"
def setup_logging(log_dir):
    log_file = log_dir / LOG_FILE_NAME; root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    logging.info("--- Semantic Indexing session started ---"); return log_file
# Chunking Function
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    chunks = textwrap.wrap(text, width=chunk_size, subsequent_indent=" " * overlap, break_long_words=False, break_on_hyphens=False)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def write_subdir_json(subdir_file_path, subdir_set):
    """Safely writes the sorted list of subdirectories to the JSON file."""
    try:
        sorted_subdirs = sorted(list(subdir_set))
        with open(subdir_file_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_subdirs, f, indent=2)
        logging.info(f"Updated subdirectory list ({len(sorted_subdirs)} entries) to {subdir_file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to write subdirectory list to {subdir_file_path}: {e}")
        return False

# --- index_directory function ---
def index_directory(source_dir, vector_db_client, embedding_model_instance, script_path, collection_name, script_dir_for_json): # <<< Added script_dir_for_json
    try:
        logging.info(f"Getting/Creating ChromaDB collection: '{collection_name}'")
        collection = vector_db_client.get_or_create_collection(name=collection_name)
        logging.info(f"Collection '{collection_name}' obtained.")
    except Exception as e:
        logging.error(f"Failed to get/create collection '{collection_name}': {e}", exc_info=True); return 0, 0, 0 # <<< Return counts only
    #Load existing subdirectories
    subdir_list_file = script_dir_for_json / f"{collection_name}_subdirs.json"
    all_encountered_subdirs = set()
    try:
        if subdir_list_file.is_file():
            with open(subdir_list_file, 'r', encoding='utf-8') as f:
                existing_subdirs = json.load(f)
                if isinstance(existing_subdirs, list):
                    all_encountered_subdirs.update(existing_subdirs)
                    logging.info(f"Loaded {len(all_encountered_subdirs)} existing subdirectories from {subdir_list_file}")
                else:
                    logging.warning(f"Content of {subdir_list_file} is not a list. Starting fresh.")
        else:
            logging.info(f"Subdirectory file {subdir_list_file} not found. Will create it.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {subdir_list_file}. Starting fresh.")
    except Exception as e:
        logging.error(f"Error reading {subdir_list_file}: {e}. Starting fresh.")

    # Get existing metadata
    try:
        existing_metadata = collection.get(include=['metadatas'])
        last_indexed_times = { meta['filename']: meta.get('last_modified', 0)
                               for meta in existing_metadata.get('metadatas', []) if meta and 'filename' in meta }
        logging.info(f"Found metadata for {len(last_indexed_times)} previously indexed files in '{collection_name}'.")
    except Exception as e:
        logging.warning(f"Could not retrieve existing metadata for '{collection_name}' or collection is empty: {e}. Will index all found files.")
        last_indexed_times = {}

    source_dir_path = Path(source_dir); files_processed, files_skipped, files_failed = 0, 0, 0
    script_dir_local = script_path.parent; vector_db_dir_abs = (script_dir_local / VECTOR_DB_PATH).resolve()
    logging.info(f"Starting recursive scan of directory: {source_dir_path}")

    for item in source_dir_path.rglob('*'):
        logging.debug(f"Found item: {item}")
        if not item.is_file(): continue
        item_abs = item.resolve()
        try:
            if item_abs.is_relative_to(vector_db_dir_abs): continue
        except ValueError: pass
        if item_abs == script_path: continue
        file_ext = item.suffix.lower()
        if file_ext not in RELEVANT_EXTENSIONS: continue

        try:
            relative_filename = item.relative_to(source_dir_path); mtime = item.stat().st_mtime
            last_indexed_time = last_indexed_times.get(str(relative_filename), 0)
            if mtime <= last_indexed_time: logging.debug(f"Skipping unchanged file: {relative_filename}"); files_skipped += 1; continue

            logging.info(f"Processing '{relative_filename}' for collection '{collection_name}'..."); files_processed += 1

            # Read file content
            file_content_str = ""
            try:
                if file_ext == ".pdf":
                    if not PYPDF2_AVAILABLE: logging.warning(f"Skipping PDF {relative_filename}: PyPDF2 library not found."); files_failed += 1; continue
                    try: reader = PdfReader(item); text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]; file_content_str = "\n\n".join(text_parts)
                    except Exception as pdf_err: logging.error(f"Failed to read PDF file {relative_filename}: {pdf_err}"); files_failed += 1; continue
                else:
                    try: file_content_str = item.read_text(encoding='utf-8')
                    except UnicodeDecodeError: logging.warning(f"UTF-8 decode failed for {relative_filename}, trying latin-1."); file_content_str = item.read_text(encoding='latin-1')
            except Exception as read_err: logging.error(f"Failed to read or extract text from file {relative_filename}: {read_err}"); files_failed += 1; continue

            chunks = split_text_into_chunks(file_content_str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if not chunks: logging.warning(f"File {relative_filename} resulted in zero processable chunks."); continue

            logging.debug(f"Generating {len(chunks)} embeddings for {relative_filename}..."); embeddings = embedding_model_instance.encode(chunks, show_progress_bar=False)
            relative_path_parts = relative_filename.parts; subdirectory = relative_path_parts[0] if len(relative_path_parts) > 1 else "root"
            logging.debug(f"Assigning subdirectory '{subdirectory}' for {relative_filename}")
            if subdirectory not in all_encountered_subdirs:
                logging.info(f"Discovered new subdirectory: '{subdirectory}'")
                all_encountered_subdirs.add(subdirectory)
                write_subdir_json(subdir_list_file, all_encountered_subdirs)
            
            ids = [f"{str(relative_filename)}_{i}" for i in range(len(chunks))]
            metadatas = [{"filename": str(relative_filename), "chunk_index": i, "last_modified": mtime, "subdirectory": subdirectory} for i in range(len(chunks))]

            try: collection.delete(where={"filename": str(relative_filename)}); logging.debug(f"Deleted old entries for {relative_filename} from '{collection_name}'")
            except Exception as del_err: logging.warning(f"Could not delete old entries for {relative_filename} from '{collection_name}' (may not exist): {del_err}")

            collection.add(ids=ids, embeddings=embeddings.tolist(), documents=chunks, metadatas=metadatas)
            logging.debug(f"Added {len(chunks)} chunks for {relative_filename} to '{collection_name}'")

        except Exception as e: logging.error(f"Failed to process file {item}: {e}", exc_info=True); files_failed += 1

    logging.info(f"Finished directory scan for collection '{collection_name}'.")
    return files_processed, files_skipped, files_failed


def update_config_file(config_path, section, key, value):
    config = configparser.ConfigParser();
    try: config.read(config_path);
    except Exception: pass
    if not config.has_section(section): config.add_section(section)
    config.set(section, key, str(value))
    try:
        with open(config_path, 'w') as configfile: config.write(configfile)
        logging.info(f"Updated '{key}' in section '[{section}]' of config file: {config_path}")
    except Exception as e: logging.error(f"Failed to update config file {config_path}: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Recursively index text files into a ChromaDB vector database, update config.ini, and save subdirectory list incrementally.", epilog=f"Example: python {script_path.name} /path/to/your/dataset -c my_collection")
    parser.add_argument("source_directory", type=str, help="The path to the directory containing files to index.")
    parser.add_argument("-c", "--collection", type=str, default=DEFAULT_COLLECTION_NAME, help=f"Name of the ChromaDB collection to index into (default: {DEFAULT_COLLECTION_NAME}).")
    args = parser.parse_args()

    source_dir = Path(args.source_directory).resolve(); collection_name_to_use = args.collection
    config_file_path = script_dir / CONFIG_FILE_NAME

    if not source_dir.is_dir(): print(f"Error: Source directory not found or is not a directory: {source_dir}"); sys.exit(1)
    log_file = setup_logging(script_dir)
    update_config_file(config_file_path, 'Paths', 'DatasetBaseDir', source_dir)

    logging.info(f"Source Directory: {source_dir}"); logging.info(f"Target Collection: '{collection_name_to_use}'")
    logging.info(f"Vector DB Path: {VECTOR_DB_PATH}"); logging.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

    if ".pdf" in RELEVANT_EXTENSIONS and not PYPDF2_AVAILABLE: logging.warning("PDF extension enabled, but PyPDF2 library not found. PDF files will be skipped.")
    vector_db_client = None; embedding_model_instance = None # Initialize
    try:
        logging.info(f"Initializing ChromaDB client from path: {VECTOR_DB_PATH}")
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        vector_db_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Fatal error during initialization: {e}", exc_info=True)
        print(f"Error during initialization, check log file: {log_file}. Exiting."); sys.exit(1)

    # --- Run Indexing ---
    start_time = time.time()
    processed, skipped, failed = 0, 0, 0
    try:
        processed, skipped, failed = index_directory(
            source_dir, vector_db_client, embedding_model_instance, script_path, collection_name_to_use, script_dir # Pass script_dir here
        )

    except KeyboardInterrupt: logging.warning("Indexing interrupted by user."); print("\nIndexing interrupted.")
    except Exception as e: logging.error("An unexpected error occurred during indexing.", exc_info=True); print(f"An unexpected error occurred, check log file: {log_file}")
    finally:
        end_time = time.time(); duration = end_time - start_time
        logging.info("--- Indexing Summary ---")
        logging.info(f"Target Collection: '{collection_name_to_use}'")
        logging.info(f"Files Processed (added/updated): {processed}")
        logging.info(f"Files Skipped (unchanged): {skipped}")
        logging.info(f"Files Failed: {failed}")
        logging.info(f"Total Duration: {duration:.2f} seconds")

    
        logging.info("--- Semantic Indexing session finished ---")
        print("\nIndexing finished. Check log for details.") # JSON file should be up-to-date now

if __name__ == "__main__":
    if not PYPDF2_AVAILABLE and ".pdf" in RELEVANT_EXTENSIONS: print("\nWARNING: PDF processing enabled but PyPDF2 not found.\nPlease install it using: pip install pypdf2\n")
    main()
