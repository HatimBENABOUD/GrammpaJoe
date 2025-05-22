import streamlit as st
import re
import nltk
from nltk.corpus import words
from symspellpy import SymSpell, Verbosity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

@st.cache_resource
def load_nltk_words():
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        st.info("NLTK 'words' corpus not found. Downloading...")
        nltk.download('words')
    base_vocab = set(words.words())
    custom_words = {"using", "streamlit", "python", "checking", "grammar", "spelling"}
    return base_vocab.union(custom_words)

@st.cache_resource
def build_symspell_dictionary(vocabulary):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for word in vocabulary:
        sym_spell.create_dictionary_entry(word, 1)
    return sym_spell

@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

VOCABULARY = load_nltk_words()
SYM_SPELL = build_symspell_dictionary(VOCABULARY)
BERT_TOKENIZER, BERT_MODEL = load_bert_model()

def simple_tokenize_normalize(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b|[.,!?;:\"'()]+", text)
    cleaned_tokens = []
    for token in tokens:
        clean_word = re.sub(r"[.,!?;:\"'()]+$", "", token)
        if clean_word:
            cleaned_tokens.append(clean_word)
        punctuation = re.findall(r"[.,!?;:\"'()]+", token)
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

def deep_learning_correction(text):
    input_text = "gec: " + text 
    input_ids = BERT_TOKENIZER.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = BERT_MODEL.generate(input_ids, max_length=512)
    corrected_text = BERT_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def check_text_logic(text, vocabulary, sym_spell_tool):
    tokens = simple_tokenize_normalize(text)
    spelling_errors = check_spelling(tokens, vocabulary, sym_spell_tool)
    grammar_errors = check_grammar(tokens)
    all_errors = sorted(spelling_errors + grammar_errors, key=lambda x: x.get('index', 0))
    corrected = deep_learning_correction(text)
    return all_errors, corrected

st.title("🧠 AI Grammar & Spell Checker (Real-Time)")
st.markdown("""
Type your text to get:
- Spelling and grammar issues (rule-based)
- AI-based sentence correction (powered by BERT)
""")

text_input = st.text_area("Start typing:", height=200, key="realtime_input")

if text_input:
    issues, ai_suggestion = check_text_logic(text_input, VOCABULARY, SYM_SPELL)
    st.subheader("Live Feedback")
    if issues:
        for issue in issues:
            st.markdown(f"**Type:** {issue['type']}")
            st.markdown(f"**Message:** {issue['message']}")
            highlighted = issue['context'][:issue['offsetInContext']] + \
                f"<span style='background-color:#de4141;padding:0 4px;border-radius:4px'><b>{issue['context'][issue['offsetInContext']:issue['offsetInContext']+issue['length']]}</b></span>" + \
                issue['context'][issue['offsetInContext']+issue['length']:]
            st.markdown(f"**In context:**<br>...{highlighted}...", unsafe_allow_html=True)
            if issue.get('suggestions'):
                st.markdown(f"**Suggestions:** {', '.join(issue['suggestions'])}")
            st.markdown("---")
    else:
        st.success("No issues found!")

    st.subheader("✨ AI Suggestion")
    st.markdown(f"**Corrected Sentence:** {ai_suggestion}")