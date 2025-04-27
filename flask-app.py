<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flask web application for querying a semantic document index (ChromaDB).
Uses a local LLM (e.g., Ollama, LM Studio) to understand user questions
and synthesize answers based on semantically relevant document chunks.
Includes conversation history and source document logging.
"""

import sys
from pathlib import Path # Ensure Path is imported
import logging # Ensure logging is imported
import requests # For making HTTP requests to the AI API
import json
import traceback # For detailed error logging
# --- START NEW IMPORTS ---
from flask import Flask, render_template, request, session # Added session
# --- END NEW IMPORTS ---


# --- Vector DB / Semantic Search Imports ---
import chromadb
from sentence_transformers import SentenceTransformer
import torch # For GPU device check

# --- Determine script directory ---
try:
    script_path = Path(__file__).resolve()
except NameError:
    script_path = Path(sys.argv[0]).resolve()
script_dir = script_path.parent
# --- Determine script directory ---


# --- Configuration ---

# == LLM Configuration ==
AI_API_ENDPOINT = "http://localhost:11434/api/generate" # Ollama default
AI_MODEL_NAME = "mistral:instruct" # Or your chosen generative model

# == Vector DB and Embedding Configuration ==
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH_STR = "F:/Code/~code/vector_db" # Absolute path used previously
VECTOR_DB_PATH = Path(VECTOR_DB_PATH_STR)
COLLECTION_NAME = "file_chunks"

# == Search Configuration ==
MAX_SEMANTIC_RESULTS = 5

# == Web App Configuration == START NEW ==
MAX_HISTORY = 10 # Number of past Q/A pairs to keep in history
# IMPORTANT: Set a secret key for session management!
# In production, use a strong, randomly generated key and keep it secret.
# You can generate one using: python -c 'import os; print(os.urandom(24))'
SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/' # Example key - REPLACE THIS!
# == Web App Configuration == END NEW ==

# == Logging Configuration ==
LOG_FILE_NAME = "web_app_query_log.txt" # Main application log
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
# --- START NEW LOG CONFIG ---
SOURCE_LOG_FILE_NAME = "response-sources.txt" # Log for sources used in responses
SOURCE_LOG_FORMAT = '%(asctime)s - %(message)s' # Simpler format for source log
# --- END NEW LOG CONFIG ---

# --- End Configuration ---

# --- Global variables ---
embedding_model_instance = None
vector_db_client = None
collection = None

# --- Flask App Setup ---
web_app = Flask(__name__)
# --- START NEW: Set Secret Key ---
web_app.secret_key = SECRET_KEY
# --- END NEW: Set Secret Key ---


# --- Configure Main Flask logging (File and Console) --- START MODIFIED BLOCK ---
log_file_path = script_dir / LOG_FILE_NAME
log_formatter = logging.Formatter(LOG_FORMAT)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

if web_app.logger.hasHandlers():
    web_app.logger.handlers.clear()
web_app.logger.addHandler(file_handler)
web_app.logger.addHandler(console_handler)
web_app.logger.setLevel(logging.INFO)

web_app.logger.info(f"--- Web Application Logging Initialized ---")
web_app.logger.info(f"Logging main activity to console and file: {log_file_path}")
# --- Configure Main Flask logging --- END MODIFIED BLOCK ---

# --- Configure Source Document Logging (File Only) --- START NEW BLOCK ---
source_log_file_path = script_dir / SOURCE_LOG_FILE_NAME
source_log_formatter = logging.Formatter(SOURCE_LOG_FORMAT)
source_file_handler = logging.FileHandler(source_log_file_path, encoding='utf-8')
source_file_handler.setFormatter(source_log_formatter)

source_logger = logging.getLogger('SourceLogger') # Create a distinct logger instance
source_logger.setLevel(logging.INFO)
source_logger.addHandler(source_file_handler)
source_logger.propagate = False # Prevent source logs from going to the main logger/console

web_app.logger.info(f"Logging response sources to file: {source_log_file_path}")
# --- Configure Source Document Logging --- END NEW BLOCK ---


# --- LLM Interaction Function ---
# ... (query_local_ai function remains unchanged) ...
def query_local_ai(prompt, is_json_mode=False):
    """Sends a prompt to the local AI API and returns the response text."""
    headers = {"Content-Type": "application/json"}
    payload = {}
    api_type = "unknown"

    # Determine API type and construct payload
    if "/api/generate" in AI_API_ENDPOINT: # Ollama style
        api_type = "ollama"
        payload = {"model": AI_MODEL_NAME, "prompt": prompt, "stream": False}
        if is_json_mode: payload["format"] = "json"
    elif "/v1/chat/completions" in AI_API_ENDPOINT: # OpenAI style
        api_type = "openai"
        payload = {
            "model": AI_MODEL_NAME or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "stream": False
        }
        if is_json_mode: payload["response_format"] = { "type": "json_object" }
    else:
        web_app.logger.warning(f"Cannot determine API type from endpoint: {AI_API_ENDPOINT}. Using generic prompt.")
        payload = {"prompt": prompt}

    if not AI_MODEL_NAME and api_type != "openai":
         web_app.logger.warning("AI_MODEL_NAME is not set. This might be required by your API.")

    web_app.logger.debug(f"Sending payload to {AI_API_ENDPOINT}: {json.dumps(payload)}")

    try:
        response = requests.post(AI_API_ENDPOINT, headers=headers, json=payload, timeout=180) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        web_app.logger.debug(f"Received response data: {response_data}")

        # Extract response based on API type
        if api_type == "ollama":
            if 'response' in response_data: return response_data['response'].strip()
            elif 'error' in response_data: web_app.logger.error(f"Ollama API Error: {response_data['error']}"); return None
        elif api_type == "openai":
            if 'choices' in response_data and response_data['choices']:
                message = response_data['choices'][0].get('message', {})
                content = message.get('content')
                if content: return content.strip()
                text_fallback = response_data['choices'][0].get('text') # Some models might use 'text'
                if text_fallback: return text_fallback.strip()
            elif 'error' in response_data: web_app.logger.error(f"OpenAI API Error: {response_data['error']}"); return None

        # Fallback extraction if structure is unknown
        web_app.logger.warning("Could not reliably extract content based on API type. Trying common keys.")
        for key in ['response', 'content', 'text']:
            if key in response_data: return response_data[key].strip()
        if isinstance(response_data, str): return response_data.strip() # Handle plain string response

        web_app.logger.error(f"Could not extract response text from AI response: {response_data}")
        return None

    except requests.exceptions.Timeout:
        web_app.logger.error(f"Timeout connecting to AI API at {AI_API_ENDPOINT}")
        return None
    except requests.exceptions.RequestException as e:
        web_app.logger.error(f"Error connecting to AI API at {AI_API_ENDPOINT}: {e}")
        return None
    except json.JSONDecodeError as e:
        web_app.logger.error(f"Error decoding JSON response from AI API: {e}")
        web_app.logger.error(f"Raw response text: {response.text[:500]}...")
        return None
    except Exception as e:
        web_app.logger.error(f"An unexpected error occurred during AI query: {e}")
        web_app.logger.error(traceback.format_exc())
        return None

def get_available_subdirectories(chroma_collection):
    """Queries the ChromaDB collection to find unique subdirectory metadata values."""
    if not chroma_collection:
        return []
    try:
        web_app.logger.info("Fetching metadata to determine available subdirectories...")
        # Get all metadata. For very large DBs, this might be slow.
        # Consider alternatives like peeking or storing this list separately if performance is an issue.
        metadata = chroma_collection.get(include=['metadatas'])
        subdirs = set()
        for meta in metadata.get('metadatas', []):
            if meta and 'subdirectory' in meta:
                subdirs.add(meta['subdirectory'])
        sorted_subdirs = sorted(list(subdirs))
        web_app.logger.info(f"Found subdirectories: {sorted_subdirs}")
        return sorted_subdirs
    except Exception as e:
        web_app.logger.error(f"Failed to get available subdirectories from metadata: {e}")
        return [] # Return empty list on error

# --- Resource Initialization ---
# ... (initialize_resources function includes GPU device setting from previous step) ...
def initialize_resources():
    """Loads the embedding model and connects to the ChromaDB database."""
    global embedding_model_instance, vector_db_client, collection
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        web_app.logger.info(f"Using device: {device}")

        web_app.logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} onto {device}")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        web_app.logger.info("Embedding model loaded.")

        web_app.logger.info(f"Initializing ChromaDB client from path: {VECTOR_DB_PATH_STR}")
        if not VECTOR_DB_PATH.exists():
             web_app.logger.error(f"ChromaDB path not found: {VECTOR_DB_PATH_STR}. Database must be created first using database-maintain.py.")
             vector_db_client = None; collection = None; embedding_model_instance = None
             return

        vector_db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH_STR)
        web_app.logger.info(f"Getting ChromaDB collection: {COLLECTION_NAME}")
        collection = vector_db_client.get_collection(name=COLLECTION_NAME)
        web_app.logger.info(f"ChromaDB collection '{COLLECTION_NAME}' obtained successfully.")
        if collection:
            available_subdirs = get_available_subdirectories(collection)
        

    except chromadb.errors.NotFoundError:
         web_app.logger.error(f"ChromaDB collection '{COLLECTION_NAME}' not found in {VECTOR_DB_PATH_STR}. Run database-maintain.py first.")
         collection = None; embedding_model_instance = None
    except ValueError as ve:
        web_app.logger.error(f"ValueError during ChromaDB initialization (other than collection not found): {ve}", exc_info=True)
        collection = None; embedding_model_instance = None
    except ImportError:
         web_app.logger.error("Error importing torch, chromadb or sentence_transformers. Make sure they are installed.")
         embedding_model_instance = None; vector_db_client = None; collection = None;
    except Exception as e:
        web_app.logger.error(f"Fatal error during resource initialization: {e}", exc_info=True)
        embedding_model_instance = None; vector_db_client = None; collection = None
    except chromadb.errors.NotFoundError:
         web_app.logger.error(f"ChromaDB collection '{COLLECTION_NAME}' not found in {VECTOR_DB_PATH_STR}. Run database-maintain.py first.")
         collection = None; embedding_model_instance = None; available_subdirs = []
    if collection is None or embedding_model_instance is None:
        embedding_model_instance = None; collection = None; vector_db_client = None; available_subdirs = []


# --- Semantic Search Function ---
# ... (search_database_vector function remains unchanged) ...
# --- Modify search_database_vector ---
def search_database_vector(user_question, n_results=MAX_SEMANTIC_RESULTS, subdirectory_filter=None): # <<< Added subdirectory_filter
    """
    Performs semantic search on the ChromaDB vector database using embeddings,
    optionally filtering by subdirectory metadata.
    """
    global collection, embedding_model_instance

    # ... (checks for user_question, collection, model remain the same) ...
    if not user_question: return []
    if collection is None or embedding_model_instance is None: return []

    try:
        web_app.logger.info(f"Generating embedding for query: '{user_question[:50]}...'")
        query_embedding = embedding_model_instance.encode([user_question], show_progress_bar=False)

        # --- START NEW: Build Where Clause ---
        where_clause = None
        if subdirectory_filter and subdirectory_filter != "all": # Assuming "all" means no filter
            where_clause = {"subdirectory": subdirectory_filter}
            web_app.logger.info(f"Applying subdirectory filter: {where_clause}")
        else:
            web_app.logger.info("No subdirectory filter applied (searching all).")
        # --- END NEW: Build Where Clause ---

        web_app.logger.info(f"Querying ChromaDB collection '{collection.name}' for {n_results} results.")
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_clause, # <<< Pass the where clause
            include=["metadatas", "documents", "distances"]
        )
        # ... (rest of result processing remains the same) ...
        raw_result_count = len(results.get('ids', [[]])[0]) if results.get('ids') else 0
        web_app.logger.info(f"ChromaDB query returned {raw_result_count} raw results.")
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]; distances = results['distances'][0]
            metadatas = results['metadatas'][0]; documents = results['documents'][0]
            for i in range(len(ids)):
                metadata = metadatas[i]
                formatted_results.append({
                    "filename": metadata.get("filename", "Unknown File"),
                    "text_chunk": documents[i],
                    "distance": distances[i],
                    "chunk_index": metadata.get("chunk_index", -1),
                    "last_modified": metadata.get("last_modified", 0),
                    "subdirectory": metadata.get("subdirectory", "N/A") # Include subdir in result if needed
                })
        web_app.logger.info(f"Formatted {len(formatted_results)} semantic search results.")
        return formatted_results

    # ... (except block remains the same) ...
    except Exception as e:
        web_app.logger.error(f"Error during semantic search for '{user_question[:50]}...': {e}")
        web_app.logger.error(traceback.format_exc())
        return []

# --- Flask Routes ---
@web_app.route('/', methods=['GET', 'POST'])
def index():
    """Handles user interaction: displays form (GET) or processes question (POST)."""
    global available_subdirs # <<< ADD THIS LINE to access the global variable

    if 'history' not in session:
        session['history'] = []

    # --- START NEW: Get selected subdir from previous request (if any) ---
    # Default to 'all' on GET or if not provided
    selected_subdir = session.get('selected_subdir', 'all')
    # --- END NEW ---

    template_data = {
        "question": "",
        "final_response": "",
        "search_results_list": [],
        "error": "",
        "history": session['history'],
        "available_subdirs": available_subdirs, # <<< This will now work
        "selected_subdir": selected_subdir
    }


    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        # --- START NEW: Get selected subdir from THIS request ---
        selected_subdir = request.form.get('selected_subdir', 'all')
        session['selected_subdir'] = selected_subdir # Store in session for next GET request
        template_data["selected_subdir"] = selected_subdir # Update template data for this response
        # --- END NEW ---
        template_data["question"] = user_question

        # ... (rest of POST logic: check question, search, format, query AI, update history) ...
        if not user_question:
            template_data["error"] = "Please enter a question."
            return render_template('index.html', **template_data)

        web_app.logger.info(f"Web user question: {user_question}")
        web_app.logger.info(f"Selected subdirectory filter: {selected_subdir}") # Log the filter

        # 1. Search Vector Database (MODIFIED CALL)
        web_app.logger.info("Searching vector database...")
        semantic_results = []
        if collection and embedding_model_instance:
             # Pass the selected subdirectory filter to the search function
             semantic_results = search_database_vector(
                 user_question,
                 n_results=MAX_SEMANTIC_RESULTS,
                 subdirectory_filter=selected_subdir # <<< Pass filter here
             )
        # ... (rest of error handling and AI response generation) ...
        else:
             template_data["error"] = "Search database is not available."
             web_app.logger.error("Search attempted while DB/model not initialized.")
             return render_template('index.html', **template_data)

        template_data["search_results_list"] = semantic_results
        final_response = None

        # 2. Format results and get final AI response (logic remains mostly the same)
        if not semantic_results:
            # ... (handle no results) ...
            web_app.logger.info("No relevant documents found via semantic search (with current filter).")
            no_results_prompt = (
                f"The user asked: \"{user_question}\". A semantic search of the document database (filtered for subdirectory: {selected_subdir}) found no relevant text chunks. "
                "Briefly inform the user that no relevant information was found in the specified scope."
            )
            final_response = query_local_ai(no_results_prompt)
            template_data["final_response"] = final_response or "No relevant documents were found in the database for your query and filter."
            if not template_data["error"]:
                 template_data["error"] = "No relevant results found."
        else:
            # ... (prepare context, query AI - maybe add filter info to prompt?) ...
            web_app.logger.info(f"Found {len(semantic_results)} relevant chunks. Preparing summary prompt...")
            results_context = ""
            # ... (build results_context as before) ...
            for i, result in enumerate(semantic_results):
                results_context += f"--- Relevant Chunk {i+1} ---\n"
                results_context += f"Source File: {result['filename']} (Subdir: {result.get('subdirectory', 'N/A')})\n" # Optionally show subdir
                results_context += f"Content: {result['text_chunk']}\n\n"

            summary_prompt = (
                f"You are a helpful assistant. Analyze the following text chunks retrieved from a document database based on semantic relevance to the user's question. The search was potentially filtered to the subdirectory '{selected_subdir}'.\n\n"
                # ... (rest of summary prompt) ...
                 f"User's Question: \"{user_question}\"\n\n"
                 "--- Retrieved Text Chunks ---\n"
                 f"{results_context}"
                 "--- End of Chunks ---\n\n"
                 "Based *only* on the information in the provided chunks, answer the user's question..." # Rest of prompt
            )
            web_app.logger.info("Asking AI to synthesize the answer from semantic results...")
            final_response = query_local_ai(summary_prompt)
            # ... (handle AI response error, log sources) ...
            if not final_response:
                template_data["error"] = "Error getting final summary from AI."
            else:
                 template_data["final_response"] = final_response
                 if semantic_results:
                     try:
                         source_files = sorted(list(set([res['filename'] for res in semantic_results])))
                         log_message = f"User Question: \"{user_question}\" | Filter: '{selected_subdir}' | Sources: {source_files}" # Add filter to log
                         source_logger.info(log_message)
                     except Exception as log_err:
                         web_app.logger.error(f"Failed to log response sources: {log_err}")

        # Update history (as before)
        if final_response:
            session['history'].append({'question': user_question, 'response': final_response})
            if len(session['history']) > MAX_HISTORY:
                session['history'].pop(0)
            session.modified = True

        template_data["history"] = session['history']

    # Render the page
    return render_template('index.html', **template_data)
# --- Main Execution ---
if __name__ == '__main__':
    web_app.logger.info("--- Starting Flask Web Application ---")

    web_app.logger.info("Initializing application resources (Embedding Model & DB Connection)...")
    initialize_resources()

    if collection is None or embedding_model_instance is None:
         web_app.logger.error("!!! CRITICAL FAILURE: Failed to initialize ChromaDB or Embedding Model. Search functionality will be disabled. Check logs above. !!!")
         print("\n!!! CRITICAL FAILURE: Could not initialize resources. Search will not work. Check logs. !!!\n", file=sys.stderr)
    else:
         web_app.logger.info("Resources initialized successfully.")

    web_app.logger.info("Starting Flask development server...")
    web_app.logger.info(f"Access the application at http://127.0.0.1:5000")
    web_app.run(debug=True, host='127.0.0.1', port=5000)

    web_app.logger.info("--- Flask Web Application Stopped ---")

=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flask web application for querying a semantic document index (ChromaDB).
Uses a local LLM (e.g., Ollama, LM Studio) to understand user questions
and synthesize answers based on semantically relevant document chunks.
Includes conversation history and source document logging.
"""

import sys
from pathlib import Path # Ensure Path is imported
import logging # Ensure logging is imported
import requests # For making HTTP requests to the AI API
import json
import traceback # For detailed error logging
# --- START NEW IMPORTS ---
from flask import Flask, render_template, request, session # Added session
# --- END NEW IMPORTS ---


# --- Vector DB / Semantic Search Imports ---
import chromadb
from sentence_transformers import SentenceTransformer
import torch # For GPU device check

# --- Determine script directory ---
try:
    script_path = Path(__file__).resolve()
except NameError:
    script_path = Path(sys.argv[0]).resolve()
script_dir = script_path.parent
# --- Determine script directory ---


# --- Configuration ---

# == LLM Configuration ==
AI_API_ENDPOINT = "http://localhost:11434/api/generate" # Ollama default
AI_MODEL_NAME = "mistral:instruct" # Or your chosen generative model

# == Vector DB and Embedding Configuration ==
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH_STR = "F:/Code/~code/vector_db" # Absolute path used previously
VECTOR_DB_PATH = Path(VECTOR_DB_PATH_STR)
COLLECTION_NAME = "file_chunks"

# == Search Configuration ==
MAX_SEMANTIC_RESULTS = 5

# == Web App Configuration == START NEW ==
MAX_HISTORY = 10 # Number of past Q/A pairs to keep in history
# IMPORTANT: Set a secret key for session management!
# In production, use a strong, randomly generated key and keep it secret.
# You can generate one using: python -c 'import os; print(os.urandom(24))'
SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/' # Example key - REPLACE THIS!
# == Web App Configuration == END NEW ==

# == Logging Configuration ==
LOG_FILE_NAME = "web_app_query_log.txt" # Main application log
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
# --- START NEW LOG CONFIG ---
SOURCE_LOG_FILE_NAME = "response-sources.txt" # Log for sources used in responses
SOURCE_LOG_FORMAT = '%(asctime)s - %(message)s' # Simpler format for source log
# --- END NEW LOG CONFIG ---

# --- End Configuration ---

# --- Global variables ---
embedding_model_instance = None
vector_db_client = None
collection = None

# --- Flask App Setup ---
web_app = Flask(__name__)
# --- START NEW: Set Secret Key ---
web_app.secret_key = SECRET_KEY
# --- END NEW: Set Secret Key ---


# --- Configure Main Flask logging (File and Console) --- START MODIFIED BLOCK ---
log_file_path = script_dir / LOG_FILE_NAME
log_formatter = logging.Formatter(LOG_FORMAT)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

if web_app.logger.hasHandlers():
    web_app.logger.handlers.clear()
web_app.logger.addHandler(file_handler)
web_app.logger.addHandler(console_handler)
web_app.logger.setLevel(logging.INFO)

web_app.logger.info(f"--- Web Application Logging Initialized ---")
web_app.logger.info(f"Logging main activity to console and file: {log_file_path}")
# --- Configure Main Flask logging --- END MODIFIED BLOCK ---

# --- Configure Source Document Logging (File Only) --- START NEW BLOCK ---
source_log_file_path = script_dir / SOURCE_LOG_FILE_NAME
source_log_formatter = logging.Formatter(SOURCE_LOG_FORMAT)
source_file_handler = logging.FileHandler(source_log_file_path, encoding='utf-8')
source_file_handler.setFormatter(source_log_formatter)

source_logger = logging.getLogger('SourceLogger') # Create a distinct logger instance
source_logger.setLevel(logging.INFO)
source_logger.addHandler(source_file_handler)
source_logger.propagate = False # Prevent source logs from going to the main logger/console

web_app.logger.info(f"Logging response sources to file: {source_log_file_path}")
# --- Configure Source Document Logging --- END NEW BLOCK ---


# --- LLM Interaction Function ---
# ... (query_local_ai function remains unchanged) ...
def query_local_ai(prompt, is_json_mode=False):
    """Sends a prompt to the local AI API and returns the response text."""
    headers = {"Content-Type": "application/json"}
    payload = {}
    api_type = "unknown"

    # Determine API type and construct payload
    if "/api/generate" in AI_API_ENDPOINT: # Ollama style
        api_type = "ollama"
        payload = {"model": AI_MODEL_NAME, "prompt": prompt, "stream": False}
        if is_json_mode: payload["format"] = "json"
    elif "/v1/chat/completions" in AI_API_ENDPOINT: # OpenAI style
        api_type = "openai"
        payload = {
            "model": AI_MODEL_NAME or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "stream": False
        }
        if is_json_mode: payload["response_format"] = { "type": "json_object" }
    else:
        web_app.logger.warning(f"Cannot determine API type from endpoint: {AI_API_ENDPOINT}. Using generic prompt.")
        payload = {"prompt": prompt}

    if not AI_MODEL_NAME and api_type != "openai":
         web_app.logger.warning("AI_MODEL_NAME is not set. This might be required by your API.")

    web_app.logger.debug(f"Sending payload to {AI_API_ENDPOINT}: {json.dumps(payload)}")

    try:
        response = requests.post(AI_API_ENDPOINT, headers=headers, json=payload, timeout=180) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        web_app.logger.debug(f"Received response data: {response_data}")

        # Extract response based on API type
        if api_type == "ollama":
            if 'response' in response_data: return response_data['response'].strip()
            elif 'error' in response_data: web_app.logger.error(f"Ollama API Error: {response_data['error']}"); return None
        elif api_type == "openai":
            if 'choices' in response_data and response_data['choices']:
                message = response_data['choices'][0].get('message', {})
                content = message.get('content')
                if content: return content.strip()
                text_fallback = response_data['choices'][0].get('text') # Some models might use 'text'
                if text_fallback: return text_fallback.strip()
            elif 'error' in response_data: web_app.logger.error(f"OpenAI API Error: {response_data['error']}"); return None

        # Fallback extraction if structure is unknown
        web_app.logger.warning("Could not reliably extract content based on API type. Trying common keys.")
        for key in ['response', 'content', 'text']:
            if key in response_data: return response_data[key].strip()
        if isinstance(response_data, str): return response_data.strip() # Handle plain string response

        web_app.logger.error(f"Could not extract response text from AI response: {response_data}")
        return None

    except requests.exceptions.Timeout:
        web_app.logger.error(f"Timeout connecting to AI API at {AI_API_ENDPOINT}")
        return None
    except requests.exceptions.RequestException as e:
        web_app.logger.error(f"Error connecting to AI API at {AI_API_ENDPOINT}: {e}")
        return None
    except json.JSONDecodeError as e:
        web_app.logger.error(f"Error decoding JSON response from AI API: {e}")
        web_app.logger.error(f"Raw response text: {response.text[:500]}...")
        return None
    except Exception as e:
        web_app.logger.error(f"An unexpected error occurred during AI query: {e}")
        web_app.logger.error(traceback.format_exc())
        return None

def get_available_subdirectories(chroma_collection):
    """Queries the ChromaDB collection to find unique subdirectory metadata values."""
    if not chroma_collection:
        return []
    try:
        web_app.logger.info("Fetching metadata to determine available subdirectories...")
        # Get all metadata. For very large DBs, this might be slow.
        # Consider alternatives like peeking or storing this list separately if performance is an issue.
        metadata = chroma_collection.get(include=['metadatas'])
        subdirs = set()
        for meta in metadata.get('metadatas', []):
            if meta and 'subdirectory' in meta:
                subdirs.add(meta['subdirectory'])
        sorted_subdirs = sorted(list(subdirs))
        web_app.logger.info(f"Found subdirectories: {sorted_subdirs}")
        return sorted_subdirs
    except Exception as e:
        web_app.logger.error(f"Failed to get available subdirectories from metadata: {e}")
        return [] # Return empty list on error

# --- Resource Initialization ---
# ... (initialize_resources function includes GPU device setting from previous step) ...
def initialize_resources():
    """Loads the embedding model and connects to the ChromaDB database."""
    global embedding_model_instance, vector_db_client, collection
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        web_app.logger.info(f"Using device: {device}")

        web_app.logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} onto {device}")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        web_app.logger.info("Embedding model loaded.")

        web_app.logger.info(f"Initializing ChromaDB client from path: {VECTOR_DB_PATH_STR}")
        if not VECTOR_DB_PATH.exists():
             web_app.logger.error(f"ChromaDB path not found: {VECTOR_DB_PATH_STR}. Database must be created first using database-maintain.py.")
             vector_db_client = None; collection = None; embedding_model_instance = None
             return

        vector_db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH_STR)
        web_app.logger.info(f"Getting ChromaDB collection: {COLLECTION_NAME}")
        collection = vector_db_client.get_collection(name=COLLECTION_NAME)
        web_app.logger.info(f"ChromaDB collection '{COLLECTION_NAME}' obtained successfully.")
        if collection:
            available_subdirs = get_available_subdirectories(collection)
        

    except chromadb.errors.NotFoundError:
         web_app.logger.error(f"ChromaDB collection '{COLLECTION_NAME}' not found in {VECTOR_DB_PATH_STR}. Run database-maintain.py first.")
         collection = None; embedding_model_instance = None
    except ValueError as ve:
        web_app.logger.error(f"ValueError during ChromaDB initialization (other than collection not found): {ve}", exc_info=True)
        collection = None; embedding_model_instance = None
    except ImportError:
         web_app.logger.error("Error importing torch, chromadb or sentence_transformers. Make sure they are installed.")
         embedding_model_instance = None; vector_db_client = None; collection = None;
    except Exception as e:
        web_app.logger.error(f"Fatal error during resource initialization: {e}", exc_info=True)
        embedding_model_instance = None; vector_db_client = None; collection = None
    except chromadb.errors.NotFoundError:
         web_app.logger.error(f"ChromaDB collection '{COLLECTION_NAME}' not found in {VECTOR_DB_PATH_STR}. Run database-maintain.py first.")
         collection = None; embedding_model_instance = None; available_subdirs = []
    if collection is None or embedding_model_instance is None:
        embedding_model_instance = None; collection = None; vector_db_client = None; available_subdirs = []


# --- Semantic Search Function ---
# ... (search_database_vector function remains unchanged) ...
# --- Modify search_database_vector ---
def search_database_vector(user_question, n_results=MAX_SEMANTIC_RESULTS, subdirectory_filter=None): # <<< Added subdirectory_filter
    """
    Performs semantic search on the ChromaDB vector database using embeddings,
    optionally filtering by subdirectory metadata.
    """
    global collection, embedding_model_instance

    # ... (checks for user_question, collection, model remain the same) ...
    if not user_question: return []
    if collection is None or embedding_model_instance is None: return []

    try:
        web_app.logger.info(f"Generating embedding for query: '{user_question[:50]}...'")
        query_embedding = embedding_model_instance.encode([user_question], show_progress_bar=False)

        # --- START NEW: Build Where Clause ---
        where_clause = None
        if subdirectory_filter and subdirectory_filter != "all": # Assuming "all" means no filter
            where_clause = {"subdirectory": subdirectory_filter}
            web_app.logger.info(f"Applying subdirectory filter: {where_clause}")
        else:
            web_app.logger.info("No subdirectory filter applied (searching all).")
        # --- END NEW: Build Where Clause ---

        web_app.logger.info(f"Querying ChromaDB collection '{collection.name}' for {n_results} results.")
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_clause, # <<< Pass the where clause
            include=["metadatas", "documents", "distances"]
        )
        # ... (rest of result processing remains the same) ...
        raw_result_count = len(results.get('ids', [[]])[0]) if results.get('ids') else 0
        web_app.logger.info(f"ChromaDB query returned {raw_result_count} raw results.")
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]; distances = results['distances'][0]
            metadatas = results['metadatas'][0]; documents = results['documents'][0]
            for i in range(len(ids)):
                metadata = metadatas[i]
                formatted_results.append({
                    "filename": metadata.get("filename", "Unknown File"),
                    "text_chunk": documents[i],
                    "distance": distances[i],
                    "chunk_index": metadata.get("chunk_index", -1),
                    "last_modified": metadata.get("last_modified", 0),
                    "subdirectory": metadata.get("subdirectory", "N/A") # Include subdir in result if needed
                })
        web_app.logger.info(f"Formatted {len(formatted_results)} semantic search results.")
        return formatted_results

    # ... (except block remains the same) ...
    except Exception as e:
        web_app.logger.error(f"Error during semantic search for '{user_question[:50]}...': {e}")
        web_app.logger.error(traceback.format_exc())
        return []

# --- Flask Routes ---
@web_app.route('/', methods=['GET', 'POST'])
def index():
    """Handles user interaction: displays form (GET) or processes question (POST)."""
    global available_subdirs # <<< ADD THIS LINE to access the global variable

    if 'history' not in session:
        session['history'] = []

    # --- START NEW: Get selected subdir from previous request (if any) ---
    # Default to 'all' on GET or if not provided
    selected_subdir = session.get('selected_subdir', 'all')
    # --- END NEW ---

    template_data = {
        "question": "",
        "final_response": "",
        "search_results_list": [],
        "error": "",
        "history": session['history'],
        "available_subdirs": available_subdirs, # <<< This will now work
        "selected_subdir": selected_subdir
    }


    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        # --- START NEW: Get selected subdir from THIS request ---
        selected_subdir = request.form.get('selected_subdir', 'all')
        session['selected_subdir'] = selected_subdir # Store in session for next GET request
        template_data["selected_subdir"] = selected_subdir # Update template data for this response
        # --- END NEW ---
        template_data["question"] = user_question

        # ... (rest of POST logic: check question, search, format, query AI, update history) ...
        if not user_question:
            template_data["error"] = "Please enter a question."
            return render_template('index.html', **template_data)

        web_app.logger.info(f"Web user question: {user_question}")
        web_app.logger.info(f"Selected subdirectory filter: {selected_subdir}") # Log the filter

        # 1. Search Vector Database (MODIFIED CALL)
        web_app.logger.info("Searching vector database...")
        semantic_results = []
        if collection and embedding_model_instance:
             # Pass the selected subdirectory filter to the search function
             semantic_results = search_database_vector(
                 user_question,
                 n_results=MAX_SEMANTIC_RESULTS,
                 subdirectory_filter=selected_subdir # <<< Pass filter here
             )
        # ... (rest of error handling and AI response generation) ...
        else:
             template_data["error"] = "Search database is not available."
             web_app.logger.error("Search attempted while DB/model not initialized.")
             return render_template('index.html', **template_data)

        template_data["search_results_list"] = semantic_results
        final_response = None

        # 2. Format results and get final AI response (logic remains mostly the same)
        if not semantic_results:
            # ... (handle no results) ...
            web_app.logger.info("No relevant documents found via semantic search (with current filter).")
            no_results_prompt = (
                f"The user asked: \"{user_question}\". A semantic search of the document database (filtered for subdirectory: {selected_subdir}) found no relevant text chunks. "
                "Briefly inform the user that no relevant information was found in the specified scope."
            )
            final_response = query_local_ai(no_results_prompt)
            template_data["final_response"] = final_response or "No relevant documents were found in the database for your query and filter."
            if not template_data["error"]:
                 template_data["error"] = "No relevant results found."
        else:
            # ... (prepare context, query AI - maybe add filter info to prompt?) ...
            web_app.logger.info(f"Found {len(semantic_results)} relevant chunks. Preparing summary prompt...")
            results_context = ""
            # ... (build results_context as before) ...
            for i, result in enumerate(semantic_results):
                results_context += f"--- Relevant Chunk {i+1} ---\n"
                results_context += f"Source File: {result['filename']} (Subdir: {result.get('subdirectory', 'N/A')})\n" # Optionally show subdir
                results_context += f"Content: {result['text_chunk']}\n\n"

            summary_prompt = (
                f"You are a helpful assistant. Analyze the following text chunks retrieved from a document database based on semantic relevance to the user's question. The search was potentially filtered to the subdirectory '{selected_subdir}'.\n\n"
                # ... (rest of summary prompt) ...
                 f"User's Question: \"{user_question}\"\n\n"
                 "--- Retrieved Text Chunks ---\n"
                 f"{results_context}"
                 "--- End of Chunks ---\n\n"
                 "Based *only* on the information in the provided chunks, answer the user's question..." # Rest of prompt
            )
            web_app.logger.info("Asking AI to synthesize the answer from semantic results...")
            final_response = query_local_ai(summary_prompt)
            # ... (handle AI response error, log sources) ...
            if not final_response:
                template_data["error"] = "Error getting final summary from AI."
            else:
                 template_data["final_response"] = final_response
                 if semantic_results:
                     try:
                         source_files = sorted(list(set([res['filename'] for res in semantic_results])))
                         log_message = f"User Question: \"{user_question}\" | Filter: '{selected_subdir}' | Sources: {source_files}" # Add filter to log
                         source_logger.info(log_message)
                     except Exception as log_err:
                         web_app.logger.error(f"Failed to log response sources: {log_err}")

        # Update history (as before)
        if final_response:
            session['history'].append({'question': user_question, 'response': final_response})
            if len(session['history']) > MAX_HISTORY:
                session['history'].pop(0)
            session.modified = True

        template_data["history"] = session['history']

    # Render the page
    return render_template('index.html', **template_data)
# --- Main Execution ---
if __name__ == '__main__':
    web_app.logger.info("--- Starting Flask Web Application ---")

    web_app.logger.info("Initializing application resources (Embedding Model & DB Connection)...")
    initialize_resources()

    if collection is None or embedding_model_instance is None:
         web_app.logger.error("!!! CRITICAL FAILURE: Failed to initialize ChromaDB or Embedding Model. Search functionality will be disabled. Check logs above. !!!")
         print("\n!!! CRITICAL FAILURE: Could not initialize resources. Search will not work. Check logs. !!!\n", file=sys.stderr)
    else:
         web_app.logger.info("Resources initialized successfully.")

    web_app.logger.info("Starting Flask development server...")
    web_app.logger.info(f"Access the application at http://127.0.0.1:5000")
    web_app.run(debug=True, host='127.0.0.1', port=5000)

    web_app.logger.info("--- Flask Web Application Stopped ---")

>>>>>>> 7b9166bce70e9cbd987bf89874279cf77562a725
