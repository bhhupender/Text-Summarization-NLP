# Project: Text Summarization using NLP

# Objective:
# Build a text summarization tool using NLP techniques to create extractive and abstractive summaries.

# Step 1: Import Required Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 2: Preprocess the Text Data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return filtered_words

# Step 3: Create a Frequency Table
def create_frequency_table(filtered_words):
    freq_table = {}
    for word in filtered_words:
        if word.lower() not in freq_table:
            freq_table[word.lower()] = 1
        else:
            freq_table[word.lower()] += 1
    return freq_table

# Step 4: Score Sentences Based on Frequency
def score_sentences(text, freq_table):
    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freq_table.keys():
                if sent not in sentence_scores:
                    sentence_scores[sent] = freq_table[word]
                else:
                    sentence_scores[sent] += freq_table[word]
    return sentence_scores

# Step 6: Generate Extractive Summary
def generate_extractive_summary(sentence_scores, threshold=25):
    summary_sentences = heapq.nlargest(
        int(len(sentence_scores) * 0.3) + 1, sentence_scores, key=sentence_scores.get
    )
    summary = ' '.join(summary_sentences)
    return summary

# Step 6: Generate Abstractive Summary using T5
def generate_abstractive_summary(text):
    # Load pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Preprocess input text for T5
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

# Step 7: User Interface with Gradio
def summarize_text(input_text):
    filtered_words = preprocess_text(input_text)
    freq_table = create_frequency_table(filtered_words)
    sentence_scores = score_sentences(input_text, freq_table)

    extractive_summary = generate_extractive_summary(sentence_scores)
    abstractive_summary = generate_abstractive_summary(input_text)

    return extractive_summary, abstractive_summary

# Step 7: Streamlit Interface
st.title("Text Summarization Tool")
st.subheader("Generate Extractive and Abstractive Summaries")

input_text = st.text_area("Enter text to summarize:", height=200)
submit_button = st.button("Submit")
if submit_button:
    if input_text.strip():
        filtered_words = preprocess_text(input_text)
        freq_table = create_frequency_table(filtered_words)
        sentence_scores = score_sentences(input_text, freq_table)

        extractive_summary = generate_extractive_summary(sentence_scores)
        abstractive_summary = generate_abstractive_summary(input_text)

        st.subheader("Extractive Summary:")
        st.write(extractive_summary)

        st.subheader("Abstractive Summary:")
        st.write(abstractive_summary)
    else:
        st.error("Please enter some text to summarize!")
