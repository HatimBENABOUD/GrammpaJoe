import streamlit as st
import re
import nltk
from nltk.corpus import words
import time
from symspellpy import SymSpell, Verbosity

@st.cache_resource
def load_nltk_words():
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        st.info("Downloading NLTK 'words' corpus...")
        nltk.download('words')
    base_vocab = set(words.words())
    custom_words = {"using", "streamlit", "python", "checking", "grammar", "spelling"}
    return base_vocab.union(custom_words)


@st.cache_resource
def build_symspell_dictionary(vocabulary):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    start_time = time.time()
    for word in vocabulary:
        sym_spell.create_dictionary_entry(word, 1)
    end_time = time.time()
    st.success(f"SymSpell dictionary built in {end_time - start_time:.2f} seconds.")
    return sym_spell

VOCABULARY = load_nltk_words()
if VOCABULARY:
    SYM_SPELL = build_symspell_dictionary(VOCABULARY)
else:
    VOCABULARY = {"the", "cat", "sat", "on", "mat", "a", "is", "in", "it", "and", "dog", "run", "runs", "jump", "jumps", "big", "small", "quick", "brown", "fox", "lazy", "dogs", "time", "of", "for", "to", "be", "have", "has", "do", "does", "go", "goes", "went", "gone"}
    SYM_SPELL = None
    st.warning("Falling back to a small hardcoded vocabulary and no SymSpell.")

def simple_tokenize_normalize(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b|[.,!?;:\"\'()]+', text)
    cleaned_tokens = []
    for token in tokens:
        clean_word = re.sub(r'[.,!?;:\"\'()]+$', '', token)
        if clean_word:
            cleaned_tokens.append(clean_word)
        punctuation = re.findall(r'[.,!?;:\"\'()]+', token)
        if punctuation:
            cleaned_tokens.extend(punctuation)
    return cleaned_tokens

def check_spelling(tokens, vocabulary, sym_spell_tool):
    errors = []
    original_text = " ".join(tokens)
    current_offset = 0
    for i, token in enumerate(tokens):
        if re.fullmatch(r'\w+', token) and token not in vocabulary:
            candidates = []
            if sym_spell_tool:
                suggestions = sym_spell_tool.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
                candidates = [s.term for s in suggestions]
            if candidates:
                try:
                    offset_in_text = original_text.index(token, current_offset)
                except ValueError:
                    offset_in_text = current_offset
                context_start = max(0, offset_in_text - 20)
                context_end = min(len(original_text), offset_in_text + len(token) + 20)
                context_text = original_text[context_start:context_end]
                offset_in_context = offset_in_text - context_start
                errors.append({
                    'tool': 'SymSpell',
                    'type': 'Spelling',
                    'token': token,
                    'index': i,
                    'message': f"Spelling mistake: '{token}'",
                    'suggestions': candidates[:3],
                    'context': context_text,
                    'offsetInContext': offset_in_context,
                    'length': len(token)
                })
            current_offset += len(token) + 1
        else:
            current_offset += len(token) + 1
    return errors

def check_grammar(tokens):
    errors = []
    original_text = " ".join(tokens)
    current_offset = 0
    for i, token in enumerate(tokens):
        try:
            offset_in_text = original_text.index(token, current_offset)
        except ValueError:
            offset_in_text = current_offset

        if token in ['he', 'she', 'it'] and i + 1 < len(tokens):
            next_token = tokens[i+1]
            if next_token in ['go', 'run', 'jump']:
                suggestion = f"{next_token}s" if next_token != 'go' else 'goes'
                errors.append({
                    'tool': 'Grammar',
                    'type': 'Subject-Verb',
                    'message': f"'{token} {next_token}' may be incorrect",
                    'suggestions': [f"{token} {suggestion}"],
                    'context': original_text[max(0, offset_in_text-20):offset_in_text+20],
                    'offsetInContext': offset_in_text - max(0, offset_in_text-20),
                    'length': len(f"{token} {next_token}")
                })

        if token == 'a' and i+1 < len(tokens):
            next_token = tokens[i+1]
            if next_token.endswith('s'):
                errors.append({
                    'tool': 'Grammar',
                    'type': 'Article-Noun Agreement',
                    'message': f"'a {next_token}' might be incorrect",
                    'suggestions': [f"some {next_token}"],
                    'context': original_text[max(0, offset_in_text-20):offset_in_text+20],
                    'offsetInContext': offset_in_text - max(0, offset_in_text-20),
                    'length': len(f"a {next_token}")
                })

        if token in ['in', 'on', 'at'] and i+1 < len(tokens):
            next_token = tokens[i+1]
            if next_token not in VOCABULARY:
                errors.append({
                    'tool': 'Grammar',
                    'type': 'Preposition Usage',
                    'message': f"'{token} {next_token}' may be incorrect",
                    'suggestions': [f"{token} the {next_token}"],
                    'context': original_text[max(0, offset_in_text-20):offset_in_text+20],
                    'offsetInContext': offset_in_text - max(0, offset_in_text-20),
                    'length': len(f"{token} {next_token}")
                })

        current_offset = offset_in_text + len(token) + 1
    return errors

def check_text_logic(text, vocabulary, sym_spell_tool):
    start_time = time.time()
    tokens = simple_tokenize_normalize(text)
    spelling_errors = check_spelling(tokens, vocabulary, sym_spell_tool)
    grammar_errors = check_grammar(tokens)
    all_errors = sorted(spelling_errors + grammar_errors, key=lambda x: x.get('index', 0))
    end_time = time.time()
    st.sidebar.write(f"Processed in {end_time - start_time:.2f} seconds")
    return all_errors

st.title("📝 Grammar & Spell Checker")
st.markdown("""
Enter text to detect common grammar and spelling issues. 
**New features:** Preposition usage check and improved highlights.
""")

text_input = st.text_area("Enter your text:", height=200)

if text_input:
    issues = check_text_logic(text_input, VOCABULARY, SYM_SPELL)
    st.subheader("Results")
    if issues:
        for issue in issues:
            st.markdown(f"**Type:** {issue['type']}")
            st.markdown(f"**Message:** {issue['message']}")
            highlighted = issue['context'][:issue['offsetInContext']] + \
                f"<span style='background-color:#fffb91;padding:0 4px;border-radius:4px'><b>{issue['context'][issue['offsetInContext']:issue['offsetInContext']+issue['length']]}</b></span>" + \
                issue['context'][issue['offsetInContext']+issue['length']:]
            st.markdown(f"**In context:**<br>...{highlighted}...", unsafe_allow_html=True)
            if issue.get('suggestions'):
                st.markdown(f"**Suggestions:** {', '.join(issue['suggestions'])}")
            st.markdown("---")
    else:
        st.success("No issues found!")