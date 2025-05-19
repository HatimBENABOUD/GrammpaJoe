import streamlit as st # Import Streamlit
import re
import nltk
from nltk.corpus import words
import time
# Import SymSpell
# You'll need to install it: pip install symspellpy streamlit
from symspellpy import SymSpell, Verbosity

# --- Download NLTK resources if not already present ---
# Streamlit runs the script from top to bottom on each interaction,
# so we should handle downloads and heavy initializations carefully.
# Using st.cache_resource helps with this.

@st.cache_resource # Cache the resource so it's only loaded once
def load_nltk_words():
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        st.info("NLTK 'words' corpus not found. Downloading...")
        nltk.download('words')
    return set(words.words())

@st.cache_resource # Cache the SymSpell dictionary building
def build_symspell_dictionary(vocabulary):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    start_time = time.time()
    for word in vocabulary:
        # Add word and its frequency (frequency 1 is fine for just a word list)
        sym_spell.create_dictionary_entry(word, 1)
    end_time = time.time()
    st.success(f"SymSpell dictionary built in {end_time - start_time:.2f} seconds.")
    return sym_spell

# Load vocabulary and build SymSpell dictionary using caching
VOCABULARY = load_nltk_words()
if VOCABULARY:
     SYM_SPELL = build_symspell_dictionary(VOCABULARY)
else:
    # Fallback to a small hardcoded list if NLTK corpus fails
    VOCABULARY = set([
        "the", "cat", "sat", "on", "mat", "a", "is", "in", "it", "and",
        "dog", "run", "runs", "jump", "jumps", "big", "small", "quick",
        "brown", "fox", "lazy", "dogs", "time", "of", "for", "to", "be",
        "have", "has", "do", "does", "go", "goes", "went", "gone"
    ])
    SYM_SPELL = None # Disable SymSpell if setup failed
    st.warning("Falling back to a small hardcoded vocabulary and no SymSpell.")


# --- 1. Introduction to NLP and Text Preprocessing ---

def simple_tokenize_normalize(text):
    """
    Basic tokenization and normalization.
    Splits text into lowercase words and handles simple punctuation.
    """
    # Convert to lowercase
    text = text.lower()
    # Use regex to find words (sequences of letters) and keep some punctuation attached
    tokens = re.findall(r'\b\w+\b|[.,!?;:"\'()]+', text)
    cleaned_tokens = []
    for token in tokens:
        clean_word = re.sub(r'[.,!?;:"\'()]+$', '', token)
        if clean_word:
            cleaned_tokens.append(clean_word)
        punctuation = re.findall(r'[.,!?;:"\'()]+', token)
        if punctuation:
            cleaned_tokens.extend(punctuation)
    return cleaned_tokens

# --- Spell Checking Logic (Using SymSpell) ---

# Levenshtein distance function is kept for conceptual understanding if needed,
# but SymSpell's lookup is used for performance.
def levenshtein_distance(s1, s2):
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

def check_spelling(tokens, vocabulary, sym_spell_tool):
    """
    Checks spelling of tokens against the vocabulary, using SymSpell for suggestions.
    Returns a list of potential spelling errors and suggestions.
    """
    errors = []
    original_text = " ".join(tokens)

    current_offset = 0
    for i, token in enumerate(tokens):
        if re.fullmatch(r'\w+', token):
            if token not in vocabulary:
                candidates = []
                if sym_spell_tool:
                    # Corrected: Changed max_edit_distance_lookup to max_edit_distance
                    suggestions = sym_spell_tool.lookup(token, Verbosity.CLOSEST,
                                                        max_edit_distance=2) # Look for candidates within edit distance 2
                    candidates = [s.term for s in suggestions]
                # No fallback to slow Levenshtein here in Streamlit for performance
                # If SymSpell is None, spell checking is effectively disabled

                if candidates: # Only add error if suggestions are found
                    try:
                        offset_in_text = original_text.index(token, current_offset)
                    except ValueError:
                        offset_in_text = current_offset

                    context_start = max(0, offset_in_text - 20)
                    context_end = min(len(original_text), offset_in_text + len(token) + 20)
                    context_text = original_text[context_start:context_end]
                    offset_in_context = offset_in_text - context_start

                    errors.append({
                        'tool': 'SymSpell Spell Checker',
                        'type': 'Spelling',
                        'token': token,
                        'index': i,
                        'message': f"Possible spelling mistake: '{token}'",
                        'suggestions': candidates[:3],
                        'context': context_text,
                        'offsetInContext': offset_in_context,
                        'length': len(token)
                    })
            current_offset += len(token) + 1
        else:
             current_offset += len(token) + 1
    return errors

# --- 2. Syntactic and Semantic Analysis (Basic Grammar Rules) ---
# --- 3. Modern Language Models (Simplified N-grams idea) ---
# --- 5. Named Entity Recognition (Very Basic Handling) ---

def check_grammar(tokens):
    """
    Performs basic rule-based grammar checks.
    Returns a list of potential grammar errors and messages.
    """
    errors = []
    sentence_start = True
    original_text = " ".join(tokens)

    current_offset = 0
    for i, token in enumerate(tokens):
        if not sentence_start and token and token[0].isupper():
            if i > 0 and tokens[i-1] not in ['.', '!', '?']:
                 pass

        if token in ['.', '!', '?']:
            sentence_start = True
        else:
            sentence_start = False

        try:
            offset_in_text = original_text.index(token, current_offset)
        except ValueError:
            offset_in_text = current_offset

        # Basic Rule 1: Check for "a" followed by a plural noun
        if token == 'a' and i + 1 < len(tokens):
            next_token = tokens[i+1]
            if next_token.endswith('s') and next_token not in VOCABULARY:
                 involved_tokens_str = f"{token} {next_token}"
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
                    'suggestions': [f"a {next_token[:-1]}", f"some {next_token}"],
                    'context': context_text,
                    'offsetInContext': offset_in_context,
                    'length': len(involved_tokens_str)
                 })

        # Basic Rule 2: Check for common subject-verb agreement errors
        if token in ['he', 'she', 'it'] and i + 1 < len(tokens):
            next_token = tokens[i+1]
            if next_token in ['go', 'run', 'jump']:
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

        # Basic Rule 3: Check for common verb tense errors
        if token in ['cat', 'dog', 'fox'] and i + 1 < len(tokens):
             next_token = tokens[i+1]
             if next_token in ['run', 'jump']:
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

        current_offset = offset_in_text + len(token) + 1

    return errors

# --- Main Checker Logic Function ---

def check_text_logic(text, vocabulary, sym_spell_tool):
    """
    Runs the basic grammar and spell checker on the input text.
    This function contains the core NLP logic.
    """
    start_time = time.time()

    tokens = simple_tokenize_normalize(text)

    spelling_errors = check_spelling(tokens, vocabulary, sym_spell_tool)
    grammar_errors = check_grammar(tokens)

    all_errors = sorted(spelling_errors + grammar_errors, key=lambda x: x.get('index', x.get('indices', [float('inf')])[0]))

    end_time = time.time()
    st.sidebar.write(f"Checking completed in {end_time - start_time:.4f} seconds.") # Display time in sidebar

    return all_errors

# --- Streamlit App Layout ---

st.title("Basic Grammar & Spell Checker")

st.write("""
This is a simple grammar and spell checker built using Python, NLTK, and SymSpell.
It demonstrates basic NLP concepts like tokenization, vocabulary lookup, edit distance
(via SymSpell), and rule-based grammar checking.
""")

text_input = st.text_area("Enter text below:", height=200, key="text_area")

# Auto-run check when text changes (Streamlit's default behavior)
# Or you could add a button: if st.button("Check Text"):
if text_input:
    issues = check_text_logic(text_input, VOCABULARY, SYM_SPELL)

    st.subheader("Results:")

    if issues:
        st.write(f"Found {len(issues)} potential issue(s):")
        # Display errors with highlighting (requires some HTML/Markdown)
        for issue in issues:
            st.markdown(f"**Type:** {issue['type']}")
            st.markdown(f"**Message:** {issue['message']}")

            # Manual highlighting for Streamlit
            context_text = issue['context']
            offset = issue['offsetInContext']
            length = issue['length']

            # Create highlighted context string using Markdown
            highlighted_context = context_text[:offset] + \
                                  f"<span style='background-color: rgba(255, 255, 0, 0.3); font-weight: bold; padding: 0 2px; border-radius: 3px;'>{context_text[offset:offset+length]}</span>" + \
                                  context_text[offset+length:]

            st.markdown(f"**In context:** *...{highlighted_context}...*", unsafe_allow_html=True)

            if issue.get('suggestions'):
                st.markdown(f"**Suggestions:** {', '.join(issue['suggestions'])}")
            st.markdown("---") # Separator

    else:
        st.success("No issues found!")