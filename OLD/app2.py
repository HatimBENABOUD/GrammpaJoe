from flask import Flask, request, jsonify, render_template
import logging
from autocorrect import Speller as AutoCorrectSpeller
from textblob import TextBlob
import nltk

nltk.download('punkt_tab')
# import gramformer # Gramformer can be heavy, ensure it's installed and models are available

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO) # Set to DEBUG for more details

# Initialize Flask app
app = Flask(__name__)

# --- Initialize NLP Tools ---

# autocorrect
try:
    autocorrect_speller = AutoCorrectSpeller(lang='en')
    logging.info("Autocorrect speller initialized successfully.")
except Exception as e:
    autocorrect_speller = None
    logging.error(f"Error initializing Autocorrect: {e}")

# TextBlob (no explicit global init needed, used on the fly)
logging.info("TextBlob is ready to be used.")

# Gramformer
# Gramformer can be resource-intensive.
# Ensure you have installed it (pip install gramformer) and downloaded necessary models.
# It typically requires PyTorch.
gf_tool = None
# try:
#     # Initialize Gramformer. Adjust model path or type as needed.
#     # By default, it might try to download a model if not found.
#     # gf_tool = gramformer.Gramformer(models=1, use_gpu=False) # models=1 for corrector
#     # logging.info("Gramformer initialized successfully.")
#     pass # Placeholder for Gramformer initialization
# except ImportError:
#     logging.warning("Gramformer library not found. Skipping Gramformer initialization.")
#     gf_tool = None
# except Exception as e:
#     logging.error(f"Error initializing Gramformer: {e}")
#     gf_tool = None

# --- Helper Functions for Text Checking ---

def check_with_autocorrect(text):
    """Checks text using autocorrect."""
    if not autocorrect_speller:
        return []
    # Autocorrect primarily corrects, doesn't just identify errors with context in the same way.
    # We can compare the original text with the corrected one or check word by word.
    words = text.split() # Simple split
    errors = []
    corrected_text_parts = []
    current_offset = 0

    for word in words:
        # Clean the word from common trailing punctuation for better checking
        # A more robust tokenizer would be better for production
        clean_word = word.strip('.,!?;:"\'()')
        if not clean_word: # if it was just punctuation
            corrected_text_parts.append(word)
            current_offset += len(word) + (1 if text[current_offset + len(word):].startswith(' ') else 0)
            continue

        corrected_word = autocorrect_speller(clean_word)
        
        # If autocorrect changed the word and it's not just a case change (e.g. 'apple' vs 'Apple')
        if corrected_word != clean_word and corrected_word.lower() != clean_word.lower():
            # Try to find the original word's position
            try:
                # This offset finding is basic and might not be perfect for repeated words
                word_offset = text.index(clean_word, current_offset)
            except ValueError:
                word_offset = current_offset # Fallback

            errors.append({
                'tool': 'Autocorrect',
                'type': 'Spelling (Autocorrected)',
                'message': f"Word '{clean_word}' auto-corrected to '{corrected_word}'",
                'context': text[max(0, word_offset-20):min(len(text), word_offset+len(clean_word)+20)],
                'offset': word_offset,
                'length': len(clean_word),
                'suggestions': [corrected_word]
            })
            # Use the corrected word in our conceptual "corrected text"
            # This part is tricky as autocorrect applies correction directly
            # For this example, we're just flagging what it *would* change.
        
        # Advance offset: find where this word ends to search for the next one
        try:
            current_offset = text.index(word, current_offset) + len(word)
        except ValueError: # Should not happen if word came from text
            current_offset += len(word)


    # Alternative: show the whole corrected sentence if it changed
    # corrected_full_text = autocorrect_speller(text)
    # if corrected_full_text != text:
    #     errors.append({
    #         'tool': 'Autocorrect',
    #         'type': 'Spelling (Full Text)',
    #         'message': 'Text was auto-corrected.',
    #         'context': text,
    #         'suggestions': [corrected_full_text]
    #     })
    return errors


def check_with_textblob(text):
    """Checks text using TextBlob."""
    blob = TextBlob(text)
    errors = []
    # TextBlob's .correct() method returns a corrected TextBlob object.
    # To find specific errors, we can iterate through words and use spellcheck.
    # This is a simplified approach.
    # Note: TextBlob's default spellchecker can be slow on very long texts.

    # For finding individual misspelled words and suggestions:
    current_offset = 0
    for word_obj in blob.words: # blob.words tokenizes
        original_word = str(word_obj)
        # TextBlob's spellcheck() returns a list of (word, confidence) tuples
        # The first one is usually the most confident correction.
        # word_obj.spellcheck() gives [(correction, confidence_score)]
        spell_check_result = word_obj.spellcheck()
        
        # Check if the most confident correction is different from the original word
        # and the original word is not, for example, an acronym or proper noun that spellcheck might flag.
        # This requires a more nuanced check in a real app (e.g. check if word is in a custom dictionary)
        if spell_check_result and spell_check_result[0][0].lower() != original_word.lower():
            # Try to find the word's position in the original text
            try:
                # This offset finding is basic
                word_offset = text.lower().index(original_word.lower(), current_offset)
            except ValueError:
                word_offset = current_offset # Fallback

            errors.append({
                'tool': 'TextBlob',
                'type': 'Spelling',
                'message': f"Possible spelling mistake: '{original_word}'",
                'context': text[max(0, word_offset-20):min(len(text), word_offset+len(original_word)+20)],
                'offset': word_offset,
                'length': len(original_word),
                'suggestions': [res[0] for res in spell_check_result[:3]] # Top 3 suggestions
            })
            current_offset = word_offset + len(original_word)
        else:
            # Advance offset even if no error
            try:
                current_offset = text.lower().index(original_word.lower(), current_offset) + len(original_word)
            except ValueError:
                 current_offset += len(original_word)


    # Optionally, include the fully corrected sentence by TextBlob
    # corrected_text = str(blob.correct())
    # if corrected_text != text:
    #     errors.append({
    #         'tool': 'TextBlob',
    #         'type': 'Spelling (Full Correction)',
    #         'message': 'TextBlob suggests this correction for the whole text.',
    #         'context': text,
    #         'suggestions': [corrected_text]
    #     })
    return errors

def check_with_gramformer(text):
    """Checks text using Gramformer."""
    if not gf_tool:
        logging.info("Gramformer tool not available or not initialized.")
        return []
    
    errors = []
    try:
        # Gramformer works best on sentences. Splitting text into sentences might be beneficial.
        # For simplicity, we'll pass the whole text, but be mindful of its limits.
        # Max length for Gramformer is often around 128-256 tokens.
        corrected_sentences = gf_tool.correct(text, max_candidates=1)
        
        # Gramformer returns a set of corrected sentences if corrections are made.
        # If no corrections, it might return the original or an empty set depending on usage.
        # The API might vary slightly based on version and how it's called.
        # This is a simplified interpretation of its output.
        
        # Assuming corrected_sentences is a list/set of strings
        # and we take the first one if available.
        if corrected_sentences:
            # Convert set to list if necessary and get the first correction
            first_correction = list(corrected_sentences)[0] if isinstance(corrected_sentences, set) else corrected_sentences[0]
            if first_correction != text:
                errors.append({
                    'tool': 'Gramformer',
                    'type': 'Grammar (Full Correction)',
                    'message': 'Gramformer suggests corrections for the text.',
                    'context': text, # Original text
                    'suggestions': [first_correction] # Corrected text
                })
    except Exception as e:
        logging.error(f"Error during Gramformer check: {e}")
        errors.append({
            'tool': 'Gramformer',
            'type': 'Error',
            'message': f"Failed to process text with Gramformer: {e}",
            'context': text,
            'suggestions': []
        })
    return errors

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    # Ensure you have an 'index.html' in a 'templates' folder.
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_text_route(): # Renamed to avoid conflict with function name
    """Handles the text checking request using multiple tools."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text_to_check = data['text']
    logging.info(f"Received text for checking (length: {len(text_to_check)})")

    all_errors = []

   


    # Autocorrect
    if autocorrect_speller:
        logging.info("Checking with Autocorrect...")
        all_errors.extend(check_with_autocorrect(text_to_check))
    else:
        logging.warning("Autocorrect not available for checking.")
        
    # TextBlob
    logging.info("Checking with TextBlob...")
    all_errors.extend(check_with_textblob(text_to_check)) # TextBlob doesn't have a global tool object like others

    # Gramformer (if enabled and initialized)
    if gf_tool:
        logging.info("Checking with Gramformer...")
        all_errors.extend(check_with_gramformer(text_to_check))
    else:
        # Add this to avoid confusion if Gramformer is commented out
        logging.info("Gramformer check skipped (tool not initialized or available).")


    if not all_errors and not autocorrect_speller:
         return jsonify({"error": "No text checking tools are available or initialized. Check server logs."}), 500


    logging.info(f"Found {len(all_errors)} potential issues from all tools.")
    return jsonify({"errors": all_errors})

# --- Main Execution ---
if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # Ensure you have a 'templates' folder with an 'index.html' file.
    # Example index.html:
    # <!DOCTYPE html>
    # <html>
    # <head><title>Text Checker</title></head>
    # <body>
    #   <h1>Text Checker</h1>
    #   <textarea id="text" rows="10" cols="50"></textarea><br>
    #   <button onclick="checkText()">Check Text</button>
    #   <div id="results"></div>
    #   <script>
    #     async function checkText() {
    #       const text = document.getElementById('text').value;
    #       const response = await fetch('/check', {
    #         method: 'POST',
    #         headers: {'Content-Type': 'application/json'},
    #         body: JSON.stringify({text: text})
    #       });
    #       const result = await response.json();
    #       const resultsDiv = document.getElementById('results');
    #       resultsDiv.innerHTML = ''; // Clear previous results
    #       if (result.error) {
    #         resultsDiv.innerHTML = `<p style="color:red;">Error: ${result.error}</p>`;
    #       } else if (result.errors && result.errors.length > 0) {
    #         let html = '<ul>';
    #         for (const err of result.errors) {
    #           html += `<li>
    #                      <strong>Tool:</strong> ${err.tool}<br>
    #                      <strong>Type:</strong> ${err.type || 'N/A'}<br>
    #                      <strong>Message:</strong> ${err.message}<br>
    #                      <strong>Context:</strong> "${err.context}"<br>
    #                      ${err.suggestions && err.suggestions.length > 0 ? `<strong>Suggestions:</strong> ${err.suggestions.join(', ')}<br>` : ''}
    #                      <small>(Offset: ${err.offset}, Length: ${err.length})</small>
    #                    </li><hr>`;
    #         }
    #         html += '</ul>';
    #         resultsDiv.innerHTML = html;
    #       } else {
    #         resultsDiv.innerHTML = '<p>No issues found.</p>';
    #       }
    #     }
    #   </script>
    # </body>
    # </html>
    app.run(debug=True)