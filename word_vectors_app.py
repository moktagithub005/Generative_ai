import streamlit as st
import gensim
import gzip
import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set page title and configuration
st.set_page_config(page_title="Word Context Explorer", layout="wide")
st.title("Word Context Explorer")
st.subheader("Find contextual meaning using Google News Word Vectors")

# Function to extract the gzip file
def extract_gz_file(gz_path, output_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return output_path

# Unzip file section
with st.expander("Unzip Word Vectors File", expanded=True):
    st.write("This will unzip the Google News word vectors file.")
    gz_path = st.text_input("Path to GZ file:", value="C:\\Users\\ajaym\\Downloads\\GoogleNews-vectors-negative300.bin.gz")
    output_path = st.text_input("Output path:", value="C:\\Users\\ajaym\\Downloads\\GoogleNews-vectors-negative300.bin")
    
    if st.button("Extract File"):
        try:
            with st.spinner("Extracting file... This may take a while"):
                extracted_path = extract_gz_file(gz_path, output_path)
                st.success(f"File successfully extracted to {extracted_path}")
        except Exception as e:
            st.error(f"Error extracting file: {str(e)}")

# Load model section
model = None
with st.expander("Load Word Vectors Model", expanded=True):
    model_path = st.text_input("Path to Word Vectors model:", value=output_path)
    
    if st.button("Load Model"):
        try:
            with st.spinner("Loading model... This may take a while"):
                st.text("Loading word vectors. This might take a few minutes...")
                model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
                st.session_state['model'] = model
                st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Use cached model if already loaded
if 'model' in st.session_state:
    model = st.session_state['model']

# CBOW implementation
def get_context_vector(words, model):
    """Calculate the average vector for a list of context words (CBOW approach)"""
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    return None

# Word context explorer section
st.header("Explore Word Context")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Find Similar Words")
        word = st.text_input("Enter a word:", value="king")
        
        if st.button("Find Similar Words"):
            if word in model:
                results = model.most_similar(word, topn=10)
                st.write(f"Words most similar to '{word}':")
                for result_word, similarity in results:
                    st.write(f"- {result_word}: {similarity:.4f}")
            else:
                st.error(f"Word '{word}' not found in vocabulary.")
    
    with col2:
        st.subheader("CBOW Context Meaning")
        context_words = st.text_area("Enter context words (comma-separated):", 
                                    value="royal, throne, crown, palace")
        
        if st.button("Find Words in this Context"):
            context_list = [w.strip() for w in context_words.split(',')]
            context_vector = get_context_vector(context_list, model)
            
            if context_vector is not None:
                # Find words similar to the context vector
                similar_words = []
                for word in model.index_to_key[:100000]:  # Limit search to most common words
                    similarity = cosine_similarity([model[word]], [context_vector])[0][0]
                    similar_words.append((word, similarity))
                
                # Sort by similarity and display top results
                similar_words.sort(key=lambda x: x[1], reverse=True)
                st.write("Words matching this context:")
                for result_word, similarity in similar_words[:10]:
                    st.write(f"- {result_word}: {similarity:.4f}")
            else:
                st.error("Could not compute context vector. Please check that your context words exist in the vocabulary.")
    
    # Word analogy section
    st.header("Word Analogies")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        word_a = st.text_input("Word A:", value="king")
    with col2:
        word_b = st.text_input("Word B:", value="man")
    with col3:
        word_c = st.text_input("Word C:", value="woman")
    
    if st.button("Solve Analogy"):
        try:
            results = model.most_similar(positive=[word_c, word_a], negative=[word_b], topn=5)
            st.write(f"If {word_a} is to {word_b} as {word_c} is to:")
            for result_word, similarity in results:
                st.write(f"- {result_word}: {similarity:.4f}")
        except KeyError as e:
            st.error(f"Error: {str(e)}. Please check that all words exist in the vocabulary.")
else:
    st.warning("Please load the word vectors model first.")

# Display model information
if model is not None:
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Vocabulary size: {len(model)}")
    st.sidebar.write(f"Vector dimension: {model.vector_size}")
    
    # Sample vocabulary
    st.sidebar.header("Sample Vocabulary")
    sample_words = list(model.index_to_key[:10])
    st.sidebar.write(", ".join(sample_words) + "...")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Start by extracting the GZ file
2. Load the word vectors model
3. Explore word similarities and context meanings
4. Try word analogies like 'king - man + woman = queen'
""")

st.sidebar.info("Note: The model is large (3-5GB) and may take several minutes to load.")