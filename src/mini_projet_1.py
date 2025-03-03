
# Imports
import os
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import evaluate
import stouputils as stp

def text_summarization(text: str, num_sentences: int = 3) -> str:
    """ Summarize text using extractive summarization with TextRank algorithm.
    
    Args:
        text (str): Input text to summarize
        num_sentences (int): Number of sentences to include in summary
        
    Returns:
        str: Summarized text containing most important sentences
    """
    # Split text into sentences
    sentences: list[str] = sent_tokenize(text)
    
    # Create TF-IDF matrix
    vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix: list = vectorizer.fit_transform(sentences).toarray()
    
    # Calculate similarity matrix using cosine similarity
    similarity_matrix: list = cosine_similarity(tfidf_matrix)
    
    # Create graph from similarity matrix
    graph: nx.Graph = nx.from_numpy_array(similarity_matrix)
    
    # Calculate scores using PageRank
    scores: dict = nx.pagerank(graph)
    
    # Get indices of top sentences
    ranked_sentences: list[tuple[int, float]] = sorted(((i, score) for i, score in scores.items()), key=lambda x: x[1], reverse=True)
    
    # Select top sentences
    selected_indices: list[int] = sorted([x[0] for x in ranked_sentences[:num_sentences]])
    
    # Combine sentences in original order
    summary: str = ' '.join([sentences[i] for i in selected_indices])
    
    return summary


@stp.measure_time()
@stp.handle_error()
def main():
    dataset: pd.DataFrame = pd.read_csv("gemma10000.csv")

    # Evaluate the model
    stp.info(f"Evaluating the model on {len(dataset)} samples")
    rouge_score: evaluate.EvaluationModule = evaluate.load("rouge")
    eval_results: dict = rouge_score.compute(predictions=dataset["rewritten_text"], references=dataset["original_text"])

    stp.info(stp.super_json_dump(eval_results))

if __name__ == "__main__":
    main()

