from flask import Flask, request, jsonify, render_template
import re
import nltk
from nltk.corpus import words
import time # Import time to measure performance

# Import SymSpell
from symspellpy import SymSpell, Verbosity

# Initialize Flask app
app = Flask(__name__)

# --- Download NLTK resources if not already present ---
# This should ideally be done once when setting up the environment
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("NLTK 'words' corpus not found. Downloading...")
    nltk.download('words')

# --- Vocabulary (using NLTK's words corpus) and SymSpell Setup ---
# If one library is allowed, NLTK's words corpus is a good source.
# Otherwise, you'd need a simple text file containing a dictionary.

# Initialize SymSpell
# max_dictionary_edit_distance: maximum edit distance for dictionary precomputation
# prefix_length: prefix length for deletes
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

try:
    # Load the NLTK words corpus
    nltk_words = words.words()
    print(f"NLTK 'words' corpus loaded. Size: {len(nltk_words)}")

    # Build the SymSpell dictionary from the NLTK words
    # This step takes some time initially but makes lookups much faster
    start_time = time.time()
    for word in nltk_words:
        # Add word and its frequency (frequency 1 is fine for just a word list)
        sym_spell.create_dictionary_entry(word, 1)
    end_time = time.time()
    print(f"SymSpell dictionary built in {end_time - start_time:.2f} seconds.")

    # Keep the NLTK vocabulary set for the grammar checker's basic lookup
    VOCABULARY = set(nltk_words)

except Exception as e:
    print(f"Error loading NLTK 'words' corpus or building SymSpell dictionary: {e}")
    print("Falling back to a small hardcoded vocabulary and no SymSpell.")
    # Fallback to a small hardcoded list if NLTK corpus or SymSpell build fails
    VOCABULARY = set([
        "the", "cat", "sat", "on", "mat", "a", "is", "in", "it", "and",
        "dog", "run", "runs", "jump", "jumps", "big", "small", "quick",
        "brown", "fox", "lazy", "dogs", "time", "of", "for", "to", "be",
        "have", "has", "do", "does", "go", "goes", "went", "gone"
    ])
    sym_spell = None # Disable SymSpell if setup failed


# --- 1. Introduction to NLP and Text Preprocessing ---

def simple_tokenize_normalize(text):
    """
    Basic tokenization and normalization.
    Splits text into lowercase words and handles simple punctuation.
    """
    # Convert to lowercase
    text = text.lower()
    # Use regex to find words (sequences of letters) and keep some punctuation attached
    # This is a very basic approach; a real tokenizer is more complex.
    tokens = re.findall(r'\b\w+\b|[.,!?;:"\'()]+', text)
    # Further clean up tokens (e.g., separate punctuation)
    cleaned_tokens = []
    for token in tokens:
        # Separate punctuation that might be attached
        clean_word = re.sub(r'[.,!?;:"\'()]+$', '', token) # Remove trailing punctuation
        if clean_word: # Add the word if it's not just punctuation
            cleaned_tokens.append(clean_word)
        # Add punctuation as separate tokens if it was part of the original token
        punctuation = re.findall(r'[.,!?;:"\'()]+', token)
        if punctuation:
            cleaned_tokens.extend(punctuation)

    return cleaned_tokens

# --- Spell Checking Logic (Now using SymSpell) ---

# We keep the Levenshtein distance function as it's a core concept,
# but SymSpell uses an optimized approach internally for candidate generation.
# The generate_spell_candidates function is replaced by SymSpell's lookup.

def check_spelling(tokens, vocabulary, sym_spell_tool):
    """
    Checks spelling of tokens against the vocabulary, using SymSpell for suggestions.
    Returns a list of potential spelling errors and suggestions.
    """
    errors = []
    original_text = " ".join(tokens) # Reconstruct a rough original text for context

    current_offset = 0
    for i, token in enumerate(tokens):
        # Only check words, not punctuation
        if re.fullmatch(r'\w+', token):
            # Check if the word is in our vocabulary (either NLTK or fallback)
            if token not in vocabulary:

                candidates = []
                if sym_spell_tool:
                    # Use SymSpell to find suggestions
                    # max_edit_distance_lookup: maximum edit distance for lookups
                    suggestions = sym_spell_tool.lookup(token, Verbosity.CLOSEST, 2) # Look for candidates within edit distance 2
                    candidates = [s.term for s in suggestions]


                # Find the offset of the token in the reconstructed text
                try:
                    offset_in_text = original_text.index(token, current_offset)
                except ValueError:
                    offset_in_text = current_offset # Fallback

                # Define context window (e.g., 20 characters before and after)
                context_start = max(0, offset_in_text - 20)
                context_end = min(len(original_text), offset_in_text + len(token) + 20)
                context_text = original_text[context_start:context_end]

                # Calculate the offset of the error *within* the context text
                offset_in_context = offset_in_text - context_start


                errors.append({
                    'tool': 'Basic Spell Checker' if not sym_spell_tool else 'SymSpell Spell Checker', # Indicate which tool was used
                    'type': 'Spelling',
                    'token': token,
                    'index': i, # Index in the token list
                    'message': f"Possible spelling mistake: '{token}'",
                    'suggestions': candidates[:3], # Suggest top 3
                    'context': context_text,
                    'offsetInContext': offset_in_context,
                    'length': len(token)
                })
            # Update current_offset to search for the next token after this one
            current_offset += len(token) + 1 # +1 for the space between tokens

        else: # If it's punctuation, just advance the offset
             current_offset += len(token) + 1


    return errors

# --- 2. Syntactic and Semantic Analysis (Basic Grammar Rules) ---
# --- 3. Modern Language Models (Simplified N-grams idea) ---
# --- 5. Named Entity Recognition (Very Basic Handling) ---

# This part remains the same as it focuses on basic grammar rules
# and does not involve the slow Levenshtein distance calculation over the full vocabulary.

def check_grammar(tokens):
    """
    Performs basic rule-based grammar checks.
    Returns a list of potential grammar errors and messages.
    """
    errors = []
    sentence_start = True # Assume the first token is a sentence start
    original_text = " ".join(tokens) # Reconstruct a rough original text for context

    current_offset = 0
    for i, token in enumerate(tokens):
        # Very basic check for capitalized words not at the start of a sentence
        if not sentence_start and token and token[0].isupper():
             # Check if the previous token was punctuation that ends a sentence
            if i > 0 and tokens[i-1] not in ['.', '!', '?']:
                 # This is a very rough heuristic for a potential proper noun or error
                 # We won't flag it as an error but note this limitation.
                 pass # Could add a flag or different type of note here

        # Reset sentence_start flag after sentence-ending punctuation
        if token in ['.', '!', '?']:
            sentence_start = True
        else:
            sentence_start = False

        # Find the offset of the token in the reconstructed text for context
        try:
            offset_in_text = original_text.index(token, current_offset)
        except ValueError:
            offset_in_text = current_offset # Fallback

        # Basic Rule 1: Check for "a" followed by a plural noun (very basic)
        if token == 'a' and i + 1 < len(tokens):
            next_token = tokens[i+1]
            # This is a very simple check; doesn't handle irregular plurals or exceptions
            # We still use the VOCABULARY set here for the grammar rule check
            if next_token.endswith('s') and next_token not in VOCABULARY: # Check if it looks like a plural
                 # Context for grammar errors might involve multiple tokens
                 involved_tokens_str = f"{token} {next_token}"
                 # Find the start offset of the first token in the error phrase
                 error_phrase_start_offset = original_text.index(token, current_offset)

                 context_start = max(0, error_phrase_start_offset - 20)
                 context_end = min(len(original_text), error_phrase_start_offset + len(involved_tokens_str) + 20)
                 context_text = original_text[context_start:context_end]
                 offset_in_context = error_phrase_start_offset - context_start


                 errors.append({
                    'tool': 'Basic Grammar Checker',
                    'type': 'Grammar (Agreement)',
                    'tokens': [token, next_token],
                    'indices': [i, i+1],
                    'message': f"Possible article-noun agreement error: 'a {next_token}'",
                    'suggestions': [f"a {next_token[:-1]}", f"some {next_token}"], # Simple suggestions
                    'context': context_text,
                    'offsetInContext': offset_in_context,
                    'length': len(involved_tokens_str)
                 })

        # Basic Rule 2: Check for common subject-verb agreement errors (very limited)
        # Example: "he go" instead of "he goes"
        if token in ['he', 'she', 'it'] and i + 1 < len(tokens):
            next_token = tokens[i+1]
            if next_token in ['go', 'run', 'jump']: # Simple base verbs
                involved_tokens_str = f"{token} {next_token}"
                error_phrase_start_offset = original_text.index(token, current_offset)

                context_start = max(0, error_phrase_start_offset - 20)
                context_end = min(len(original_text), error_phrase_start_offset + len(involved_tokens_str) + 20)
                context_text = original_text[context_start:context_end]
                offset_in_context = error_phrase_start_offset - context_start

                errors.append({
                    'tool': 'Basic Grammar Checker',
                    'type': 'Grammar (Subject-Verb)',
                    'tokens': [token, next_token],
                    'indices': [i, i+1],
                    'message': f"Possible subject-verb agreement error: '{token} {next_token}'",
                    'suggestions': [f"{token} {next_token}es" if next_token == 'go' else f"{token} {next_token}s"],
                    'context': context_text,
                    'offsetInContext': offset_in_context,
                    'length': len(involved_tokens_str)
                })

        # Basic Rule 3: Check for common verb tense errors (very limited)
        # Example: "the cat run" instead of "the cat runs" (simple present)
        if token in ['cat', 'dog', 'fox'] and i + 1 < len(tokens): # Simple singular subjects
             next_token = tokens[i+1]
             if next_token in ['run', 'jump']: # Simple base verbs
                involved_tokens_str = f"{token} {next_token}"
                error_phrase_start_offset = original_text.index(token, current_offset)

                context_start = max(0, error_phrase_start_offset - 20)
                context_end = min(len(original_text), error_phrase_start_offset + len(involved_tokens_str) + 20)
                context_text = original_text[context_start:context_end]
                offset_in_context = error_phrase_start_offset - context_start

                errors.append({
                    'tool': 'Basic Grammar Checker',
                    'type': 'Grammar (Verb Tense)',
                    'tokens': [token, next_token],
                    'indices': [i, i+1],
                    'message': f"Possible verb tense error: '{token} {next_token}'",
                    'suggestions': [f"{token} {next_token}s"],
                    'context': context_text,
                    'offsetInContext': offset_in_context,
                    'length': len(involved_tokens_str)
                  })

        # Update current_offset to search for the next token after this one
        current_offset = offset_in_text + len(token) + 1 # +1 for the space between tokens


    return errors


# --- 6. Text Generation (Suggestions are basic text generation) ---
# Suggestions are generated within the spell_checking and grammar_checking functions.

# --- 7. Evaluation (Needs a separate process and test data) ---
# Evaluation would involve comparing the output of this checker on a test set
# with known correct text or annotations and calculating metrics like precision/recall.
# This code doesn't include the evaluation framework itself, but the output
# format is suitable for evaluation.

# --- Main Checker Function ---

def check_text_logic(text):
    """
    Runs the basic grammar and spell checker on the input text.
    This function contains the core NLP logic.
    """
    start_time = time.time() # Start timing

    # Simple tokenization and normalization
    tokens = simple_tokenize_normalize(text)
    # print(f"Tokens: {tokens}") # For debugging

    # Perform checks
    spelling_errors = check_spelling(tokens, VOCABULARY, sym_spell) # Pass sym_spell tool
    grammar_errors = check_grammar(tokens)

    # Combine and sort errors by their position in the original text (approximated)
    # Sorting by the index of the first token involved in the error
    all_errors = sorted(spelling_errors + grammar_errors, key=lambda x: x.get('index', x.get('indices', [float('inf')])[0]))

    end_time = time.time() # End timing
    print(f"Text checking completed in {end_time - start_time:.4f} seconds.") # Print elapsed time

    return all_errors

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Assumes index.html is in a 'templates' folder
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_text_route():
    """Handles the text checking request."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text_to_check = data['text']
    # print(f"Received text for checking: {text_to_check}") # For debugging

    # Run the core checking logic
    issues = check_text_logic(text_to_check)

    # Return results as JSON
    return jsonify({"errors": issues})

# --- Main Execution ---
if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # Ensure you have a 'templates' folder with an 'index.html' file.
    # You can run this Flask app using: python your_script_name.py
    app.run(debug=True) # debug=True is useful during development
