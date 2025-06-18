# -*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import os
import pathlib
import base64
from datetime import datetime, timezone # Added timezone
import sys # For Loguru sink
# Need to import uuid for request_id
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Flask utils
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from time import time
import pandas as pd
# models
from apsislipiscanv1.ocr import ImageOCR  
from database.manager import SQLDatabaseManager, handle_user_strict_previous_login # Assuming these are correctly defined
from database.utils import config_loader

# --- Loguru Setup ---
from loguru import logger
logger.remove() # Remove default console logger
log_file_path = os.path.join(os.path.dirname(__file__), "logs", "ocr_app_{time:YYYY-MM-DD}.log")
# Ensure log directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)

logger.add(
    sys.stderr, # Log to console
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO" 
)
logger.add(
    log_file_path, # Log to file
    rotation="00:00",  # New file daily at midnight
    retention="7 days", # Keep logs for 7 days
    compression="zip",  # Compress old log files
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG", # Log DEBUG and above to file
    enqueue=True # Asynchronous logging for performance
)
# --- End Loguru Setup ---


# Define a flask app
app = Flask(__name__)
# initialize ocr
try:
    ocr = ImageOCR()
    logger.info("ImageOCR initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize ImageOCR: {e}")
    # Depending on severity, you might want to sys.exit() here
    ocr = None # Or handle appropriately

try:
    db_config_path = os.path.join(os.path.dirname(__file__), "configs", "database.yaml")
    db_config = config_loader(db_config_path) # Ensure config_loader handles path correctly
    db = SQLDatabaseManager(db_config)
    logger.info("SQLDatabaseManager initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize SQLDatabaseManager: {e}")
    db = None # Or handle appropriately

app.config['CORS_HEADERS'] = 'Content-Type'

# This function seems unused in the provided API endpoints, but kept if needed elsewhere
def convert_and_save(b64_string, file_path):
    try:
        with open(file_path, "wb") as fh:
            fh.write(base64.decodebytes(b64_string.encode()))
        logger.debug(f"Successfully decoded and saved base64 string to {file_path}")
    except Exception as e:
        logger.error(f"Error in convert_and_save for {file_path}: {e}")


def consttruct_error(msg, etype, msg_code, details, suggestion=""):
    exec_error = {"code": msg_code,
                  "type": etype,
                  "message": msg,
                  "details": details,
                  "suggestion": suggestion}
    return exec_error


def post_process_words_data(words):
    if not words: # Handle empty words list
        logger.warning("post_process_words_data received empty words list.")
        return ""
    try:
        df = pd.DataFrame(words)
        if not all(col in df.columns for col in ['text', 'line_num', 'word_num']):
            logger.error(f"Missing required columns in words data for post-processing. Columns: {df.columns.tolist()}")
            return "Error: OCR data format incorrect."

        df = df[['text', 'line_num', 'word_num']]
        lines = []
        # Ensure line_num and word_num are numeric and handle potential NaNs if data is malformed
        df['line_num'] = pd.to_numeric(df['line_num'], errors='coerce')
        df['word_num'] = pd.to_numeric(df['word_num'], errors='coerce')
        df.dropna(subset=['line_num', 'word_num'], inplace=True)
        df['line_num'] = df['line_num'].astype(int)
        df['word_num'] = df['word_num'].astype(int)

        _lines = sorted([_line for _line in df.line_num.unique()])
        for line_num_val in _lines:
            ldf = df.loc[df.line_num == line_num_val].copy() # Use .copy() to avoid SettingWithCopyWarning
            # ldf.reset_index(drop=True, inplace=True) # Not strictly necessary if only using .iloc after sort
            ldf.sort_values('word_num', inplace=True)
            _ltext = ' '.join(ldf['text'].astype(str).tolist()) # More robust way to join
            lines.append(_ltext)
        text = "\n".join(lines)
        return text
    except Exception as e:
        logger.exception(f"Error during post_process_words_data: {e}")
        return "Error during text post-processing."


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def ocr_process(): # Renamed from index to be more specific
    if request.method == 'POST':
        request_id = str(uuid.uuid4()) # For tracing
        logger.info(f"Request ID: {request_id} - Received OCR request at {datetime.now(timezone.utc)}")
        
        if ocr is None:
            logger.error(f"Request ID: {request_id} - OCR service not available.")
            return jsonify({"status": "failed", "error": consttruct_error("OCR Service Unavailable", "SERVICE_UNAVAILABLE", "503", "The OCR engine is not initialized.", "Please contact support.")}), 503

        log_details = {"req-id": request_id, "req-time": datetime.now(timezone.utc).isoformat()}
        
        try:
            save_start = time()
            basepath = os.path.dirname(__file__)
            image_dir = os.path.join(basepath, "images")
            os.makedirs(image_dir, exist_ok=True) # Ensure images directory exists

            if "file" not in request.files:
                logger.warning(f"Request ID: {request_id} - No file part in the request.")
                return jsonify({"status": "failed", "error": consttruct_error("No file provided.", "MISSING_FILE", "400", "File part is missing in the request.", "Please include a file in your request.")}), 400

            f = request.files['file']
            if f.filename == '':
                logger.warning(f"Request ID: {request_id} - No selected file (empty filename).")
                return jsonify({"status": "failed", "error": consttruct_error("No file selected.", "EMPTY_FILENAME", "400", "No file was selected for upload.", "Please select a file to upload.")}), 400

            filename = secure_filename(f.filename)
            file_path = os.path.join(image_dir, filename)
            
            file_ext = pathlib.Path(filename).suffix.lower()
            if file_ext not in [".jpg", ".png", ".jpeg"]:
                log_details["error"] = f"received file-extension:{file_ext}"
                logger.warning(f"Request ID: {request_id} - Invalid image format: {file_ext}. Details: {log_details}")
                return jsonify({"status": "failed", "error": consttruct_error("Image format not valid.", "INVALID_IMAGE_FORMAT", "400", f"Received file-extension:{file_ext}", "Please send .jpg, .png, or .jpeg image files")}), 400
            
            f.save(file_path)
            log_details["file-name"] = filename
            log_details["file-save-time"] = round(time() - save_start, 2)
            logger.info(f"Request ID: {request_id} - File '{filename}' saved to '{file_path}'. Save time: {log_details['file-save-time']}s.")

            proc_start = time()
            words = ocr(file_path, retun_text_data_only=True) # Assuming ocr() can handle path
            text = post_process_words_data(words)
            log_details["ocr-processing-time"] = round(time() - proc_start, 2)
            
            logger.info(f"Request ID: {request_id} - OCR processing complete. Time: {log_details['ocr-processing-time']}s. Log details: {log_details}")
            
            # Consider removing the file after processing if it's temporary
            # try:
            #     os.remove(file_path)
            #     logger.debug(f"Request ID: {request_id} - Successfully removed temporary file: {file_path}")
            # except OSError as e:
            #     logger.error(f"Request ID: {request_id} - Error removing temporary file {file_path}: {e}")

            return jsonify({"detailed_data": words, "plain_text": text, "status": "success"})#, "request_id": request_id})
            
        except Exception as e:
            logger.exception(f"Request ID: {request_id} - An unexpected error occurred during OCR processing: {e}. Log details: {log_details}")
            return jsonify({"status": "failed", "error": consttruct_error("Internal server error during OCR.", "INTERNAL_SERVER_ERROR", "500", str(e), "Please try again with a different image or contact support."), "request_id": request_id}), 500
    
    # GET request to this endpoint
    logger.info(f"Received GET request on / (OCR endpoint).")
    return jsonify({"message": "This is the OCR processing endpoint. Please use POST to submit an image.", "status": "info"}), 200


@app.route('/user', methods=['POST']) # Typically user creation/login is POST only
@cross_origin()
def user_session_manager(): # Renamed from index for clarity
    request_id = str(uuid.uuid4()) # For tracing
    logger.info(f"Request ID: {request_id} - Received /user request at {datetime.now(timezone.utc)}")

    if db is None:
        logger.error(f"Request ID: {request_id} - Database service not available.")
        return jsonify({"status": "failed", "error": consttruct_error("Database Service Unavailable", "SERVICE_UNAVAILABLE", "503", "The database connection is not initialized.", "Please contact support.")}), 503

    if not request.is_json:
        logger.warning(f"Request ID: {request_id} - Request body is not JSON.")
        return jsonify({"status": "failed", "error": consttruct_error("Invalid request format.", "BAD_REQUEST", "400", "Request body must be JSON.", "Ensure Content-Type is application/json.")}), 400

    try:
        data = request.get_json()
        user_email = data.get("user_email")
        user_name = data.get("user_name")

        if not user_email or not user_name:
            logger.warning(f"Request ID: {request_id} - Missing user_email or user_name in request. Data: {data}")
            missing_fields = []
            if not user_email: missing_fields.append("user_email")
            if not user_name: missing_fields.append("user_name")
            return jsonify({"status": "failed", "error": consttruct_error("Missing required fields.", "BAD_REQUEST", "400", f"Required fields missing: {', '.join(missing_fields)}", "Please provide both user_email and user_name.")}), 400

        logger.info(f"Request ID: {request_id} - Processing user: email='{user_email}', name='{user_name}'")

        # Call the user handling logic from your database manager
        user_info = handle_user_strict_previous_login(db, user_email, user_name)
        
        # handle_user_strict_previous_login might return an error dict itself
        if user_info.get("user_type") == "error":
             logger.error(f"Request ID: {request_id} - Error from handle_user_strict_previous_login: {user_info.get('error')}")
             return jsonify({"status":"failed", "error": consttruct_error(user_info.get('error', "User processing failed"), "USER_PROCESSING_ERROR", "500", "Failed during user creation or update logic.", "Please try again or contact support."), "request_id": request_id}), 500


        # Convert datetime objects to ISO format string for JSON serialization
        if 'registered' in user_info and isinstance(user_info['registered'], datetime):
            user_info['registered'] = user_info['registered'].isoformat()
        if 'last_login' in user_info and user_info['last_login'] is not None and isinstance(user_info['last_login'], datetime):
            user_info['last_login'] = user_info['last_login'].isoformat()
        
        logger.info(f"Request ID: {request_id} - Successfully processed user. Type: {user_info.get('user_type')}. User ID: {user_info.get('user_id')}")
        return jsonify({"status": "success", "data": user_info, "request_id": request_id})

    except Exception as e:
        logger.exception(f"Request ID: {request_id} - An unexpected error occurred in /user endpoint: {e}")
        return jsonify({"status": "failed", "error": consttruct_error("Internal server error.", "INTERNAL_SERVER_ERROR", "500", str(e), "An unexpected error occurred. Please try again or contact support."), "request_id": request_id}), 500


if __name__ == '__main__':
    host = "0.0.0.0"
    port = 3040
    debug_mode = False
    
    logger.info(f"Starting Flask app. Host: {host}, Port: {port}, Debug: {debug_mode}")
    app.run(debug=debug_mode, host=host, port=port)