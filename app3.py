import re
import nltk
from nltk.corpus import words # Using re for simple tokenization/punctuation handling

nltk.download('words')
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

# If one library is allowed, you could load NLTK's words corpus here.
VOCABULARY = set(words.words())

# --- Spell Checking Logic ---

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two words.
    This is a common technique for measuring string similarity.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_distance = distances[i1]
            else:
                new_distance = 1 + min((distances[i1], distances[i1 + 1], new_distances[-1]))
            new_distances.append(new_distance)
        distances = new_distances
    return distances[-1]

def generate_spell_candidates(word, vocabulary, max_distance=1):
    """
    Generate potential spelling corrections for a word
    using Levenshtein distance.
    """
    candidates = []
    for vocab_word in vocabulary:
        if levenshtein_distance(word, vocab_word) <= max_distance:
            candidates.append(vocab_word)
    # Simple ranking: sort by distance (closer is better), then alphabetically
    candidates.sort(key=lambda x: (levenshtein_distance(word, x), x))
    return candidates

def check_spelling(tokens, vocabulary):
    """
    Checks spelling of tokens against the vocabulary.
    Returns a list of potential spelling errors and suggestions.
    """
    errors = []
    for i, token in enumerate(tokens):
        # Only check words, not punctuation
        if re.fullmatch(r'\w+', token):
            if token not in vocabulary:
                candidates = generate_spell_candidates(token, vocabulary)
                errors.append({
                    'type': 'Spelling',
                    'token': token,
                    'index': i, # Index in the token list
                    'message': f"Possible spelling mistake: '{token}'",
                    'suggestions': candidates[:3] # Suggest top 3
                })
    return errors

# --- 2. Syntactic and Semantic Analysis (Basic Grammar Rules) ---
# --- 3. Modern Language Models (Simplified N-grams idea) ---
# --- 5. Named Entity Recognition (Very Basic Handling) ---

# This is a very simplified rule-based grammar checker.
# It looks for specific patterns of tokens that are often incorrect.
# A real grammar checker would involve POS tagging and parsing.
# We'll also add a very basic check for capitalized words that aren't at the start
# of a sentence as a rudimentary NER-like check to avoid flagging proper nouns.

def check_grammar(tokens):
    """
    Performs basic rule-based grammar checks.
    Returns a list of potential grammar errors and messages.
    """
    errors = []
    sentence_start = True # Assume the first token is a sentence start

    for i, token in enumerate(tokens):
        # Very basic check for capitalized words not at the start of a sentence
        if not sentence_start and token[0].isupper():
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

        # Basic Rule 1: Check for "a" followed by a plural noun (very basic)
        if token == 'a' and i + 1 < len(tokens):
            next_token = tokens[i+1]
            # This is a very simple check; doesn't handle irregular plurals
            if next_token.endswith('s') and next_token not in VOCABULARY: # Check if it looks like a plural
                 errors.append({
                    'type': 'Grammar (Agreement)',
                    'tokens': [token, next_token],
                    'indices': [i, i+1],
                    'message': f"Possible article-noun agreement error: 'a {next_token}'",
                    'suggestions': [f"a {next_token[:-1]}", f"some {next_token}"] # Simple suggestions
                 })

        # Basic Rule 2: Check for common subject-verb agreement errors (very limited)
        # Example: "he go" instead of "he goes"
        if token in ['he', 'she', 'it'] and i + 1 < len(tokens):
            next_token = tokens[i+1]
            if next_token in ['go', 'run', 'jump']: # Simple base verbs
                errors.append({
                    'type': 'Grammar (Subject-Verb)',
                    'tokens': [token, next_token],
                    'indices': [i, i+1],
                    'message': f"Possible subject-verb agreement error: '{token} {next_token}'",
                    'suggestions': [f"{token} {next_token}es" if next_token == 'go' else f"{token} {next_token}s"]
                })

        # Basic Rule 3: Check for common verb tense errors (very limited)
        # Example: "the cat run" instead of "the cat runs" (simple present)
        if token in ['cat', 'dog', 'fox'] and i + 1 < len(tokens): # Simple singular subjects
             next_token = tokens[i+1]
             if next_token in ['run', 'jump']: # Simple base verbs
                  errors.append({
                    'type': 'Grammar (Verb Tense)',
                    'tokens': [token, next_token],
                    'indices': [i, i+1],
                    'message': f"Possible verb tense error: '{token} {next_token}'",
                    'suggestions': [f"{token} {next_token}s"]
                  })


    return errors


# --- 6. Text Generation (Suggestions are basic text generation) ---
# Suggestions are generated within the spell_checking and grammar_checking functions.

# --- 7. Evaluation (Needs a separate process and test data) ---
# Evaluation would involve comparing the output of this checker on a test set
# with known correct text or annotations and calculating metrics like precision/recall.
# This code doesn't include the evaluation framework itself, but the output
# format is suitable for evaluation.

# --- Main Checker Function ---

def check_text(text):
    """
    Runs the basic grammar and spell checker on the input text.
    """
    tokens = simple_tokenize_normalize(text)
    print(f"Tokens: {tokens}") # For debugging

    spelling_errors = check_spelling(tokens, VOCABULARY)
    grammar_errors = check_grammar(tokens)

    # Combine and sort errors by their position in the text
    all_errors = sorted(spelling_errors + grammar_errors, key=lambda x: x.get('index', x.get('indices', [float('inf')])[0]))

    return all_errors

# --- Example Usage ---
if __name__ == "__main__":
    test_text = "The cat sat on teh mat. a apples is red. He go to the store. The dogs run fast."
    print(f"Checking text: '{test_text}'\n")

    issues = check_text(test_text)

    if issues:
        print("Potential Issues Found:")
        for issue in issues:
            print(f"- Type: {issue['type']}")
            print(f"  Message: {issue['message']}")
            # Display context using original text (requires re-tokenization or offset tracking)
            # For simplicity here, we'll just show the relevant tokens
            print(f"  Tokens involved: {issue.get('tokens', [issue.get('token')])}")
            if issue.get('suggestions'):
                print(f"  Suggestions: {', '.join(issue['suggestions'])}")
            print("-" * 20)
    else:
        print("No issues found.")

    print("\n--- Another Example ---")
    test_text_2 = "A big brown fox jumps over the lazy doggs."
    print(f"Checking text: '{test_text_2}'\n")
    issues_2 = check_text(test_text_2)
    if issues_2:
        print("Potential Issues Found:")
        for issue in issues_2:
            print(f"- Type: {issue['type']}")
            print(f"  Message: {issue['message']}")
            print(f"  Tokens involved: {issue.get('tokens', [issue.get('token')])}")
            if issue.get('suggestions'):
                print(f"  Suggestions: {', '.join(issue['suggestions'])}")
            print("-" * 20)
    else:
        print("No issues found.")