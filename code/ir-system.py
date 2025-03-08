import re
import math
import numpy as n
import os
from collections import defaultdict
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Initialize text processing tools
text_stemmer = PorterStemmer()
text_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the constants
OUTPUT_DIR = "output"
DOCUMENTS_PATH = r"C:\Users\afrat\OneDrive\Desktop\mos\Mechanics_of_Search_ASS2\dataset\cranfield-trec-dataset-main\cran.all.1400.xml"
QUERIES_PATH = r"C:\Users\afrat\OneDrive\Desktop\mos\Mechanics_of_Search_ASS2\dataset\cranfield-trec-dataset-main\cran.qry.xml"

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)


def extract_xml_with_regex(file_path, element_tag, content_tag):
    """
    Extracts specific content from an XML file using regular expressions.
    Given an element tag and content tag, it finds the relevant data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Regex pattern to extract content between specified tags in the xml document.
    pattern = re.compile(rf"<{element_tag}.*?>.*?<{content_tag}>(.*?)</{content_tag}>.*?</{element_tag}>", re.DOTALL)
    matches = pattern.findall(content)
    return [match.strip() for match in matches]


def clean_and_split_text(text):
    """
    Cleans the input text by removing punctuation and then splits it into individual words (tokens).
    """
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    return cleaned_text.split() 


def process_text_tokens(tokens):
    """
    Processes each token by removing stopwords, lemmatizing, and stemming.
    It helps reduce words to their base forms for more accurate matching.
    """
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemma = text_lemmatizer.lemmatize(token)  
            stemmed = text_stemmer.stem(lemma) 
            processed_tokens.append(stemmed)
    return processed_tokens


def prepare_text(text):
    """
    Prepares text by cleaning, tokenizing, and processing tokens (removes stopwords, lemmatizing, and stemming).
    """
    tokens = clean_and_split_text(text)  
    return process_text_tokens(tokens) 


def build_inverted_index(processed_docs):
    """
    Builds an inverted index from processed documents.
    The index maps terms to lists of document IDs and their term frequencies.
    """
    inverted_index = defaultdict(list)
    doc_frequency = defaultdict(int)

    for doc_id, doc in enumerate(processed_docs):
        term_freq = defaultdict(int)
        for term in doc:
            term_freq[term] += 1  

        for term, freq in term_freq.items():
            inverted_index[term].append((doc_id, freq))  
            doc_frequency[term] += 1 

    return inverted_index, doc_frequency


def tf_idf(query_terms, inverted_index, doc_frequency, num_docs):
    """
    Computes the TF-IDF vector for a query using the inverted index.
    TF-IDF helps weigh terms based on their frequency in a document and their rarity across all documents.
    """
    query_tf_idf = defaultdict(float)

    for term in query_terms:
        if term in inverted_index:
            df = doc_frequency[term]
            idf = math.log(num_docs / (df + 1))  
            query_tf_idf[term] = (query_terms.count(term) / len(query_terms)) * idf 

    return query_tf_idf


def cosine_similarity(query_tf_idf, inverted_index, doc_lengths, num_docs):
    """
    Computes cosine similarity between the query and all documents using their TF-IDF vectors.
    Cosine similarity measures how similar two vectors are, ranging from 0 to 1.
    """
    scores = [0.0] * num_docs

    for term, tf_idf in query_tf_idf.items():
        if term in inverted_index:
            for doc_id, term_freq in inverted_index[term]:
                scores[doc_id] += tf_idf * (term_freq / doc_lengths[doc_id])  

    return scores


def document_lengths(inverted_index, num_docs):
    """
    Precomputes the lengths of documents needed for cosine similarity calculations.
    The length is calculated as the square root of the sum of squared term frequencies in a document.
    """
    doc_lengths = [0.0] * num_docs

    for term, postings in inverted_index.items():
        for doc_id, term_freq in postings:
            doc_lengths[doc_id] += (term_freq ** 2) 

    return [math.sqrt(length) for length in doc_lengths]  


def score_bm25(query, vocabulary, doc_frequency, total_docs, processed_docs, inverted_index, k1=2.0, b=0.9):
    """
    Computes BM25 scores for a query.
    BM25 is a probabilistic retrieval model that helps rank documents based on term frequency and document length.
    """

    avg_doc_len = n.mean([len(doc) for doc in processed_docs])

    scores = [0.0] * total_docs

    for term in query:
        if term in inverted_index:
            df = doc_frequency[term]  
            idf = math.log((total_docs - df + 0.5) / (df + 0.5))  

            for doc_id, term_freq in inverted_index[term]:
                doc_len = len(processed_docs[doc_id])  
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
                scores[doc_id] += idf * (numerator / denominator)

    return scores


def gen_trec_results(ranked_docs, scores, query_id, run_id):
    """
    Generates output in TREC format for ranked documents, which can be used for evaluation.
    TREC format includes the query ID, document ID, rank, score, and the run ID.
    """
    output_lines = []
    for rank, doc_idx in enumerate(ranked_docs, start=1):
        score = scores[doc_idx]
        if n.isnan(score):
            score = 0.0 
        output_lines.append(f"{query_id} Q0 {doc_idx + 1} {rank} {score:.6f} {run_id}")
    return output_lines


def bert_embeddings(texts, batch_size=32, max_length=128):
    """
    Computes DistilBERT embeddings for a list of texts.
    It processes texts in batches for efficiency and uses the DistilBERT model to get the embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  
        embeddings.extend(batch_embeddings)
    return embeddings


def create_bert_embeddings(processed_docs, output_file="document_embeddings.npy"):
    """
    Precomputes BERT embeddings for all documents and saves them to a file.
    This allows us to use precomputed embeddings for similarity calculations instead of recomputing each time.
    """
    doc_texts = [' '.join(doc) for doc in processed_docs]  
    doc_embeddings = bert_embeddings(doc_texts)  
    n.save(output_file, doc_embeddings)  
    print(f"Document embeddings saved to {output_file}")


def load_document_embeddings(input_file="document_embeddings.npy"):
    """
    Loads precomputed document embeddings from a file.
    This saves time by not having to recompute the embeddings every time.
    """
    return n.load(input_file)  


def bert_similarity(query, document_embeddings):
    """
    Computes the similarity between a query and precomputed document embeddings using cosine similarity.
    The higher the cosine similarity, the more relevant the document is to the query.
    """
    query_embedding = bert_embeddings([' '.join(query)])[0]  
    scores = n.zeros(len(document_embeddings))
    for i, doc_embedding in enumerate(document_embeddings):
        dot_product = n.dot(query_embedding, doc_embedding.T).item()  
        norm_query = n.linalg.norm(query_embedding)
        norm_doc = n.linalg.norm(doc_embedding)
        if norm_query == 0 or norm_doc == 0:
            scores[i] = 0.0 
        else:
            scores[i] = dot_product / (norm_query * norm_doc) 
    return scores


def output_file_generator(output_lines, output_file):
    """
    Writes the TREC output lines to a file for evaluation.
    Each line represents a ranked document for a query.
    """
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    print(f"Output file '{output_file}' generated successfully.")


def main():
    """Main function to execute the retrieval system."""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)  
    documents = extract_xml_with_regex(DOCUMENTS_PATH, 'doc', 'text')
    queries = extract_xml_with_regex(QUERIES_PATH, 'top', 'title')
    processed_docs = [prepare_text(doc) for doc in documents]
    processed_queries = [prepare_text(query) for query in queries]
    inverted_index, doc_frequency = build_inverted_index(processed_docs)
    total_docs = len(processed_docs)
    doc_lengths = document_lengths(inverted_index, total_docs)
    vsm_output = []
    for query_id, query in enumerate(processed_queries, start=1):
        query_tf_idf = tf_idf(query, inverted_index, doc_frequency, total_docs)
        scores = cosine_similarity(query_tf_idf, inverted_index, doc_lengths, total_docs)
        ranked_docs = n.argsort(scores)[::-1]
        vsm_output.extend(gen_trec_results(ranked_docs, scores, query_id, 'VSM'))
    output_file_generator(vsm_output, os.path.join(OUTPUT_DIR, "vsm_results.txt"))
    bm25_output = []
    for query_id, query in enumerate(processed_queries, start=1):
        scores = score_bm25(query, list(inverted_index.keys()), doc_frequency, total_docs, processed_docs, inverted_index)
        ranked_docs = n.argsort(scores)[::-1]
        bm25_output.extend(gen_trec_results(ranked_docs, scores, query_id, 'BM25'))
    output_file_generator(bm25_output, os.path.join(OUTPUT_DIR, "bm25_results.txt"))
    output_lines_bert = []
    if not os.path.exists("document_embeddings.npy"):
        print("Precomputing document embeddings...")
        create_bert_embeddings(processed_docs, output_file="document_embeddings.npy")
    document_embeddings = load_document_embeddings(input_file="document_embeddings.npy")
    for query_id, query in enumerate(processed_queries, start=1):
        scores = bert_similarity(query, document_embeddings)
        ranked_docs = n.argsort(scores)[::-1]
        output_lines_bert.extend(gen_trec_results(ranked_docs, scores, query_id, 'DISTILBERT'))
    output_file_generator(output_lines_bert, os.path.join(OUTPUT_DIR, "distilbert_results.txt"))


if __name__ == "__main__":
    main()
