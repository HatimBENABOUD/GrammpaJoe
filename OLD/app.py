from flask import Flask, request, jsonify, render_template
import language_tool_python
import logging

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO) # Set to DEBUG for more details

# Initialize Flask app
app = Flask(__name__)

# Initialize LanguageTool
# This might take a moment on the first run as it downloads the language model.
# Specify the language (e.g., 'en-US', 'en-GB', 'fr', 'de', etc.)
try:
    # Automatically detect language if not specified, but specifying is better
    # Using 'en-US' as an example
    tool = language_tool_python.LanguageTool('en-US')
    logging.info("LanguageTool initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing LanguageTool: {e}")
    logging.error("Please ensure Java is installed and accessible.")
    # You might want to exit or handle this more gracefully in a real app
    tool = None

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_text():
    """Handles the text checking request."""
    if not tool:
        return jsonify({"error": "LanguageTool could not be initialized. Check server logs and Java installation."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text_to_check = data['text']
    logging.info(f"Received text for checking (length: {len(text_to_check)})")

    try:
        # Perform the check
        matches = tool.check(text_to_check)

        # Format the results
        errors = []
        for match in matches:
            errors.append({
                'message': match.message,
                'context': match.context,
                'offset': match.offset,
                'length': match.errorLength,
                'suggestions': match.replacements[:5] # Limit suggestions
            })
        logging.info(f"Found {len(errors)} potential issues.")
        return jsonify({"errors": errors})

    except Exception as e:
        logging.error(f"Error during text check: {e}")
        return jsonify({"error": "An internal error occurred during checking."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Runs the Flask development server.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True) # debug=True provides auto-reloading and more error details
