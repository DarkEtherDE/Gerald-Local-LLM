import sys
from pathlib import Path
import logging
import requests
import json
import traceback
import os
import csv
from io import StringIO
import configparser
from flask import Flask, render_template, request, session, send_from_directory, abort
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, session, send_from_directory, abort
from flask_session import Session
try:
    script_path = Path(__file__).resolve()
except NameError:
    script_path = Path(sys.argv[0]).resolve()
script_dir = script_path.parent

web_app = Flask(__name__)
LOG_FILE_NAME = "query.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
SOURCE_LOG_FILE_NAME = "sources.txt"
SOURCE_LOG_FORMAT = '%(asctime)s - %(message)s'

log_file_path = script_dir / LOG_FILE_NAME
log_formatter = logging.Formatter(LOG_FORMAT)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
if web_app.logger.hasHandlers(): web_app.logger.handlers.clear()
web_app.logger.addHandler(file_handler); web_app.logger.addHandler(console_handler)
web_app.logger.setLevel(logging.INFO)
web_app.logger.info(f"--- Web Application Logging Initialized ---") # Logger is now ready
web_app.logger.info(f"Logging main activity to console and file: {log_file_path}")
source_log_file_path = script_dir / SOURCE_LOG_FILE_NAME
source_log_formatter = logging.Formatter(SOURCE_LOG_FORMAT)
source_file_handler = logging.FileHandler(source_log_file_path, encoding='utf-8')
source_file_handler.setFormatter(source_log_formatter)
source_logger = logging.getLogger('SourceLogger'); source_logger.setLevel(logging.INFO)
source_logger.addHandler(source_file_handler); source_logger.propagate = False
web_app.logger.info(f"Logging response sources to file: {source_log_file_path}")

#Read from config.ini
config = configparser.ConfigParser()
config_file_path = script_dir / 'config.ini'
web_app.logger.info(f"Attempting to read config file from: {config_file_path}")
read_ok = config.read(config_file_path)
if not read_ok:
    web_app.logger.error(f"FAILED to read configuration file: {config_file_path}")
else:
    web_app.logger.info(f"Successfully read configuration file: {config_file_path}")

AI_API_ENDPOINT = config.get('OtherSettings', 'AI_API_ENDPOINT', fallback="http://localhost:11434/api/generate")
AI_MODEL_NAME = config.get('OtherSettings', 'AI_MODEL_NAME', fallback="mistral:instruct")

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH_STR = str(script_dir / "vector_db")
VECTOR_DB_PATH = Path(VECTOR_DB_PATH_STR)
DEFAULT_COLLECTION_NAME = "file_chunks"

dataset_dir_config = config.get('Paths', 'DatasetBaseDir', fallback=None)
web_app.logger.info(f"Value read for '[Paths]/DatasetBaseDir' from config: {dataset_dir_config}")

if dataset_dir_config:
    DATASET_BASE_DIR = Path(dataset_dir_config).resolve()
    web_app.logger.info(f"Using DatasetBaseDir from config file: {DATASET_BASE_DIR}")
else:
    DATASET_BASE_DIR = None
    web_app.logger.warning(f"'DatasetBaseDir' not found in '[Paths]' section of '{config_file_path}' or config file not read. PDF/File viewing will be disabled.")
ALLOWED_VIEW_EXTENSIONS = {'.txt', '.csv', '.pdf'}

#Number of relevant chunks to show
MAX_SEMANTIC_RESULTS = 5

MAX_HISTORY = 10
SECRET_KEY = config.get('OtherSettings', 'FLASK_SECRET_KEY', fallback='_w3lcomeHome_ChangeMe_In_Config!')

web_app.config["SECRET_KEY"] = SECRET_KEY 
web_app.config["SESSION_TYPE"] = "filesystem" 
#where session files are stored (defaults to 'flask_session' directory)
web_app.config["SESSION_FILE_DIR"] = str(script_dir / "flask_app_sessions")
web_app.config["SESSION_PERMANENT"] = False 
web_app.config["SESSION_USE_SIGNER"] = True 
web_app.config["SESSION_FILE_THRESHOLD"] = 30 # Max number of session files before cleanup (optional)
server_session = Session(web_app)

# --- Global variables ---
embedding_model_instance = None
vector_db_client = None
available_collections = []

def query_local_ai(prompt, is_json_mode=False):
    """Sends a prompt to the local AI API and returns the response text."""
    headers = {"Content-Type": "application/json"}; payload = {}; api_type = "unknown"
    if "/api/generate" in AI_API_ENDPOINT: api_type = "ollama"; payload = {"model": AI_MODEL_NAME, "prompt": prompt, "stream": False};
    elif "/v1/chat/completions" in AI_API_ENDPOINT: api_type = "openai"; payload = {"model": AI_MODEL_NAME or "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "stream": False};
    else: web_app.logger.warning(f"Cannot determine API type from endpoint: {AI_API_ENDPOINT}. Using generic prompt."); payload = {"prompt": prompt}
    if not AI_MODEL_NAME and api_type != "openai": web_app.logger.warning("AI_MODEL_NAME is not set.")
    web_app.logger.debug(f"Sending payload to {AI_API_ENDPOINT}: {json.dumps(payload)}")
    try:
        response = requests.post(AI_API_ENDPOINT, headers=headers, json=payload, timeout=180); response.raise_for_status()
        response_data = response.json(); web_app.logger.debug(f"Received response data: {response_data}")
        if api_type == "ollama":
            if 'response' in response_data: return response_data['response'].strip()
            elif 'error' in response_data: web_app.logger.error(f"Ollama API Error: {response_data['error']}"); return None
        elif api_type == "openai":
            if 'choices' in response_data and response_data['choices']:
                message = response_data['choices'][0].get('message', {}); content = message.get('content')
                if content: return content.strip()
                text_fallback = response_data['choices'][0].get('text');
                if text_fallback: return text_fallback.strip()
            elif 'error' in response_data: web_app.logger.error(f"OpenAI API Error: {response_data['error']}"); return None
        web_app.logger.warning("Could not reliably extract content. Trying common keys.");
        for key in ['response', 'content', 'text']:
            if key in response_data: return response_data[key].strip()
        if isinstance(response_data, str): return response_data.strip()
        web_app.logger.error(f"Could not extract response text from AI response: {response_data}"); return None
    except requests.exceptions.Timeout: web_app.logger.error(f"Timeout connecting to AI API at {AI_API_ENDPOINT}"); return None
    except requests.exceptions.RequestException as e: web_app.logger.error(f"Error connecting to AI API at {AI_API_ENDPOINT}: {e}"); return None
    except json.JSONDecodeError as e: web_app.logger.error(f"Error decoding JSON response from AI API: {e}"); web_app.logger.error(f"Raw response text: {response.text[:500]}..."); return None
    except Exception as e: web_app.logger.error(f"An unexpected error occurred during AI query: {e}"); web_app.logger.error(traceback.format_exc()); return None


# --- Helper Functions ---
def get_available_collections(client):
    """Gets a list of collection objects from the ChromaDB client."""
    if not client: return []
    try:
        collections = client.list_collections()
        web_app.logger.info(f"Found collections: {[col.name for col in collections]}")
        return collections
    except Exception as e:
        web_app.logger.error(f"Failed to list collections: {e}")
        return []

def load_available_subdirectories(collection_name):
    """Loads the list of unique subdirectories from a JSON file generated by builder.py."""
    subdir_file = script_dir / f"{collection_name}_subdirs.json"
    web_app.logger.info(f"Attempting to load subdirectories for collection '{collection_name}' from: {subdir_file}")
    if subdir_file.is_file():
        try:
            with open(subdir_file, 'r', encoding='utf-8') as f:
                subdirs = json.load(f)
            if isinstance(subdirs, list):
                web_app.logger.info(f"Successfully loaded {len(subdirs)} subdirectories for '{collection_name}' from {subdir_file}")
                return subdirs
            else:
                web_app.logger.error(f"Error: Content of {subdir_file} is not a JSON list.")
                return []
        except json.JSONDecodeError as e:
            web_app.logger.error(f"Error decoding JSON from {subdir_file}: {e}")
            return []
        except Exception as e:
            web_app.logger.error(f"Failed to read or parse subdirectories file {subdir_file}: {e}", exc_info=True)
            return []
    else:
        web_app.logger.warning(f"Subdirectory list file not found: {subdir_file}. Subdirectory filtering will be unavailable or incomplete. Run builder.py to generate it.")
        return []

def initialize_resources():
    """Loads the embedding model and connects to the ChromaDB client."""
    global embedding_model_instance, vector_db_client, available_collections
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'; web_app.logger.info(f"Using device: {device}")
        web_app.logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} onto {device}")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device); web_app.logger.info("Embedding model loaded.")
        web_app.logger.info(f"Initializing ChromaDB client from path: {VECTOR_DB_PATH_STR}")
        if not VECTOR_DB_PATH.exists() or not VECTOR_DB_PATH.is_dir():
             web_app.logger.error(f"ChromaDB path not found or not a directory: {VECTOR_DB_PATH_STR}. Database must be created first using builder.py.")
             vector_db_client = None; embedding_model_instance = None; available_collections = []
             return
        vector_db_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH_STR)); web_app.logger.info("ChromaDB client initialized successfully.")
        available_collections = get_available_collections(vector_db_client)
        if DATASET_BASE_DIR and DATASET_BASE_DIR.is_dir():
            web_app.logger.info(f"Dataset base directory for file viewing: {DATASET_BASE_DIR}")
        else:
            web_app.logger.warning(f"Dataset base directory not configured or invalid: {DATASET_BASE_DIR}. File viewing routes will fail.")
    except ImportError: web_app.logger.error("Error importing packages."); embedding_model_instance = None; vector_db_client = None; available_collections = []
    except Exception as e: web_app.logger.error(f"Fatal error during resource initialization: {e}", exc_info=True); embedding_model_instance = None; vector_db_client = None; available_collections = []
    if vector_db_client is None or embedding_model_instance is None:
        embedding_model_instance = None; vector_db_client = None; available_collections = []


def search_database_vector(collection_object, user_question, n_results=MAX_SEMANTIC_RESULTS, subdirectory_filter=None):
    """Performs semantic search on the provided ChromaDB collection object."""
    global embedding_model_instance
    if not user_question: return []
    if collection_object is None or embedding_model_instance is None: web_app.logger.error("Search called with invalid collection object or missing embedding model."); return []
    try:
        web_app.logger.info(f"Generating embedding for query: '{user_question[:50]}...'"); query_embedding = embedding_model_instance.encode([user_question], show_progress_bar=False)
        where_clause = None
        if subdirectory_filter and subdirectory_filter != "all": where_clause = {"subdirectory": subdirectory_filter}; web_app.logger.info(f"Applying subdirectory filter: {where_clause}")
        else: web_app.logger.info("No subdirectory filter applied (searching all).")
        web_app.logger.info(f"Querying ChromaDB collection '{collection_object.name}' for {n_results} results.")
        results = collection_object.query(query_embeddings=query_embedding.tolist(), n_results=n_results, where=where_clause, include=["metadatas", "documents", "distances"])
        raw_result_count = len(results.get('ids', [[]])[0]) if results.get('ids') else 0; web_app.logger.info(f"ChromaDB query returned {raw_result_count} raw results.")
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            ids=results['ids'][0]; distances=results['distances'][0]; metadatas=results['metadatas'][0]; documents=results['documents'][0]
            for i in range(len(ids)): metadata = metadatas[i]; formatted_results.append({"filename": metadata.get("filename", "Unknown File"), "text_chunk": documents[i], "distance": distances[i], "chunk_index": metadata.get("chunk_index", -1), "last_modified": metadata.get("last_modified", 0), "subdirectory": metadata.get("subdirectory", "N/A")})
        web_app.logger.info(f"Formatted {len(formatted_results)} semantic search results.")
        return formatted_results
    except Exception as e: web_app.logger.error(f"Error during semantic search in '{collection_object.name}': {e}"); web_app.logger.error(traceback.format_exc()); return []

@web_app.route('/', methods=['GET', 'POST'])
def index():
    """Handles user interaction: displays form (GET) or processes question (POST)."""
    global vector_db_client, embedding_model_instance, available_collections # Use global collections list
    query = session.get('query', '')
    results = []
    error_message = None
    if 'history' not in session: session['history'] = []
    selected_collection_name = session.get('selected_collection', DEFAULT_COLLECTION_NAME)
    selected_subdir = session.get('selected_subdir', 'all')
    collection_names = [col.name for col in available_collections] # Get names from global list
    current_available_subdirs = []
    current_collection_obj = None
    if vector_db_client:
        current_collection_obj_list = [col for col in available_collections if col.name == selected_collection_name]
        if current_collection_obj_list:
            current_collection_obj = current_collection_obj_list[0]
            current_available_subdirs = load_available_subdirectories(selected_collection_name)
        else:
            web_app.logger.warning(f"Selected collection '{selected_collection_name}' not found in available list. Resetting.")
            if DEFAULT_COLLECTION_NAME in collection_names:
                selected_collection_name = DEFAULT_COLLECTION_NAME
                default_collection_obj_list = [col for col in available_collections if col.name == DEFAULT_COLLECTION_NAME]
                if default_collection_obj_list:
                    current_collection_obj = default_collection_obj_list[0]
                    current_available_subdirs = load_available_subdirectories(DEFAULT_COLLECTION_NAME)
            else: 
                selected_collection_name = collection_names[0] if collection_names else None
                if selected_collection_name:
                     first_collection_obj_list = [col for col in available_collections if col.name == selected_collection_name]
                     if first_collection_obj_list:
                         current_collection_obj = first_collection_obj_list[0]
                         current_available_subdirs = load_available_subdirectories(selected_collection_name)
            session['selected_collection'] = selected_collection_name # Update session
            session['selected_subdir'] = 'all'
            selected_subdir = 'all'
    if selected_subdir != 'all' and selected_subdir not in current_available_subdirs:
        web_app.logger.warning(f"Selected subdirectory '{selected_subdir}' not valid for collection '{selected_collection_name}'. Resetting to 'all'.")
        selected_subdir = 'all'
        session['selected_subdir'] = 'all'
    if request.method == 'POST':
        web_app.logger.info("Processing POST request...")
        new_collection_name = request.form.get('selected_collection')
        if new_collection_name and new_collection_name != selected_collection_name:
            if new_collection_name in collection_names:
                selected_collection_name = new_collection_name
                session['selected_collection'] = selected_collection_name
                session['selected_subdir'] = 'all' 
                selected_subdir = 'all'
                web_app.logger.info(f"Collection changed to: {selected_collection_name}")
                current_collection_obj_list = [col for col in available_collections if col.name == selected_collection_name]
                if current_collection_obj_list:
                    current_collection_obj = current_collection_obj_list[0]
                    current_available_subdirs = load_available_subdirectories(selected_collection_name)
                else:
                    web_app.logger.error(f"Collection '{selected_collection_name}' selected but not found in available list during POST.")
                    error_message = f"Error accessing newly selected collection '{selected_collection_name}'."
                    current_collection_obj = None; current_available_subdirs = []
            else:
                web_app.logger.warning(f"POST request tried to switch to invalid collection: {new_collection_name}")
                error_message = f"Invalid collection selected: {new_collection_name}"
        new_subdir = request.form.get('selected_subdir') # Name from form dropdown
        if new_subdir and new_subdir != selected_subdir:
             if new_subdir == 'all' or new_subdir in current_available_subdirs:
                 selected_subdir = new_subdir
                 session['selected_subdir'] = selected_subdir
                 web_app.logger.info(f"Subdirectory filter changed to: {selected_subdir}")
             else:
                 web_app.logger.warning(f"POST request tried to switch to invalid subdirectory '{new_subdir}' for collection '{selected_collection_name}'. Keeping '{selected_subdir}'.")
        query = request.form.get('query', '').strip()
        session['query'] = query # Store query in session
        if not query:
            error_message = "Please enter a query."
        elif not current_collection_obj:
            error_message = f"Cannot perform query: Collection '{selected_collection_name}' is not accessible."
            web_app.logger.error(f"Query submitted but collection '{selected_collection_name}' object is not available.")
        elif not embedding_model_instance:
             error_message = "Cannot perform query: Embedding model not loaded."
             web_app.logger.error("Query submitted but embedding model is not available.")
        else:
            web_app.logger.info(f"Performing search in '{selected_collection_name}' (subdir: '{selected_subdir}') for query: '{query}'")
            try:
                search_results = search_database_vector(
                    current_collection_obj, query, n_results=MAX_SEMANTIC_RESULTS, subdirectory_filter=selected_subdir
                )
                final_response = None
                if not search_results:
                    web_app.logger.info("No relevant documents found via semantic search (with current filter).")
                    no_results_prompt = (
                        f"The user asked: \"{query}\". A semantic search of the document collection '{selected_collection_name}' (filtered for subdirectory: {selected_subdir}) found no relevant text chunks. "
                        "Briefly inform the user that no relevant information was found in the specified scope."
                    )
                    final_response = query_local_ai(no_results_prompt)
                    results = []
                    if not error_message: error_message = "No relevant results found." # Set error if not already set
                else:
                    results = search_results
                    web_app.logger.info(f"Found {len(results)} relevant chunks. Preparing summary prompt...")
                    results_context = ""
                    for i, result in enumerate(results):
                        results_context += f"--- Relevant Chunk {i+1} ---\nSource File: {result['filename']} (Subdir: {result.get('subdirectory', 'N/A')})\nContent: {result['text_chunk']}\n\n"
                    summary_prompt = (
                        f"You are a helpful assistant. Analyze the following text chunks retrieved from the '{selected_collection_name}' document collection based on semantic relevance to the user's question. The search was potentially filtered to the subdirectory '{selected_subdir}'.\n\n"
                        f"User's Question: \"{query}\"\n\n"
                        "--- Retrieved Text Chunks ---\n"
                        f"{results_context}"
                        "--- End of Chunks ---\n\n"
                        "Based *only* on the information in the provided chunks, answer the user's question. Be concise and directly address the question. Do not mention the chunk numbers or the process of searching. If the chunks do not contain the answer, state that the information is not available in the retrieved documents."
                    )
                    web_app.logger.info("Asking AI to synthesize the answer from semantic results...")
                    final_response = query_local_ai(summary_prompt)
                    if not final_response:
                        error_message = "Error getting final summary from AI."
                    else:
                        try:
                            source_files = sorted(list(set([res['filename'] for res in results])))
                            log_message = f"User Question: \"{query}\" | Collection: '{selected_collection_name}' | Filter: '{selected_subdir}' | Sources: {source_files}"
                            source_logger.info(log_message)
                        except Exception as log_err:
                            web_app.logger.error(f"Failed to log response sources: {log_err}")
                if final_response:
                    session['history'].append({'question': query, 'response': final_response})
                    if len(session['history']) > MAX_HISTORY:
                        session['history'].pop(0)
                    session.modified = True
            except Exception as e:
                error_message = f"An error occurred during search or AI call: {e}"
                web_app.logger.error(f"Error during search/AI processing: {e}", exc_info=True)
                results = [] # Clear results on error
    template_data = {
        "query": query,
        "results": results, # Contains the list of search result dictionaries
        "error": error_message,
        "history": session['history'],
        "available_collections": collection_names,
        "selected_collection": selected_collection_name,
        "available_subdirs": current_available_subdirs,
        "selected_subdir": selected_subdir,
        "DATASET_BASE_DIR": DATASET_BASE_DIR
    }
    if request.method == 'POST' and 'final_response' in locals() and final_response:
        template_data['final_response'] = final_response
    return render_template('index.html', **template_data)

@web_app.route('/view_pdf/<path:relative_filename>')
def view_pdf(relative_filename):
    """Securely serves original PDF files from the DATASET_BASE_DIR."""
    web_app.logger.info(f"Request to view PDF: {relative_filename}")
    if not DATASET_BASE_DIR or not DATASET_BASE_DIR.is_dir():
        web_app.logger.error("PDF view failed: Dataset directory not configured or invalid.")
        abort(500, description="Dataset directory not configured.")
    try:
        safe_relative_path = Path(os.path.normpath(relative_filename))
        if '..' in safe_relative_path.parts:
             web_app.logger.warning(f"Directory traversal attempt blocked for PDF: {relative_filename}")
             abort(403)
        requested_path = DATASET_BASE_DIR.joinpath(safe_relative_path).resolve()
        if not requested_path.is_relative_to(DATASET_BASE_DIR.resolve()):
            web_app.logger.warning(f"Path resolution outside base directory blocked for PDF: {relative_filename} -> {requested_path}")
            abort(403)
        if not requested_path.is_file():
            web_app.logger.error(f"PDF file not found at resolved path: {requested_path}")
            abort(404)
        if requested_path.suffix.lower() != '.pdf':
            web_app.logger.error(f"Invalid file type requested for PDF view: {requested_path.suffix}")
            abort(403, description="Invalid file type.")
        directory = requested_path.parent
        filename = requested_path.name
        web_app.logger.info(f"Serving PDF file: {filename} from {directory}")
        return send_from_directory(directory, filename, as_attachment=False) # Display inline
    except Exception as e:
        web_app.logger.error(f"Unexpected error viewing PDF {relative_filename}: {e}", exc_info=True)
        abort(500)

@web_app.route('/view_file/<path:relative_filename>')
def view_file(relative_filename):
    web_app.logger.info(f"Request to view file: {relative_filename}")
    if not DATASET_BASE_DIR or not DATASET_BASE_DIR.is_dir():
        web_app.logger.error("File view failed: Dataset directory not configured or invalid.")
        abort(500, description="Dataset directory not configured.")
    try:
        safe_relative_path = Path(os.path.normpath(relative_filename))
        if '..' in safe_relative_path.parts:
             web_app.logger.warning(f"Directory traversal attempt blocked for file: {relative_filename}")
             abort(403)
        requested_path = DATASET_BASE_DIR.joinpath(safe_relative_path).resolve()
        if not requested_path.is_relative_to(DATASET_BASE_DIR.resolve()):
             web_app.logger.warning(f"Path resolution outside base directory blocked for file: {relative_filename} -> {requested_path}")
             abort(403)
        if not requested_path.is_file():
            web_app.logger.error(f"File not found at resolved path: {requested_path}")
            abort(404)
        file_ext = requested_path.suffix.lower()
        if file_ext not in ALLOWED_VIEW_EXTENSIONS or file_ext == '.pdf': # Check against allowed non-PDF types
             web_app.logger.error(f"Invalid file type requested for file view: {file_ext}")
             abort(403, description="Invalid file type for this view.")
        content = None; csv_data = None; file_type = None
        if file_ext == '.txt':
            file_type = 'txt'
            try:
                try: content = requested_path.read_text(encoding='utf-8')
                except UnicodeDecodeError: content = requested_path.read_text(encoding='latin-1')
                web_app.logger.info(f"Serving TXT content for: {relative_filename}")
            except Exception as e: web_app.logger.error(f"Error reading TXT {relative_filename}: {e}"); abort(500)
        elif file_ext == '.csv':
            file_type = 'csv'
            try:
                try: file_content_str = requested_path.read_text(encoding='utf-8')
                except UnicodeDecodeError: file_content_str = requested_path.read_text(encoding='latin-1')
                csv_file_like = StringIO(file_content_str)
                csv_data = list(csv.reader(csv_file_like))
                web_app.logger.info(f"Serving CSV content for: {relative_filename}")
            except Exception as e: web_app.logger.error(f"Error reading/parsing CSV {relative_filename}: {e}"); abort(500)

        return render_template('view_file.html', filename=relative_filename, content=content, csv_data=csv_data, file_type=file_type)
    except Exception as e:
        web_app.logger.error(f"Error viewing file {relative_filename}: {e}", exc_info=True)
        abort(500)

if __name__ == '__main__':
    web_app.logger.info("--- Starting Flask Web Application ---")
    web_app.logger.info("Initializing application resources...")
    initialize_resources()

    if vector_db_client is None or embedding_model_instance is None:
         web_app.logger.critical("!!! CRITICAL FAILURE: Failed to initialize ChromaDB client or Embedding Model. Search functionality will be disabled. Check logs above. !!!")
         print("\n!!! CRITICAL FAILURE: Could not initialize resources. Search will not work. Check query.log. !!!\n", file=sys.stderr)
    else:
         web_app.logger.info("Resources initialized successfully.")
         web_app.logger.info(f"Available collections found: {[col.name for col in available_collections]}")

    web_app.logger.info("Starting Flask development server...")
    web_app.logger.info(f"Access the application at http://127.0.0.1:5000")
    # Use host='0.0.0.0' to make accessible on the network
    web_app.run(debug=True, host='0.0.0.0', port=5000)

    web_app.logger.info("--- Flask Web Application Stopped ---")