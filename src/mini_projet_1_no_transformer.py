
# Imports
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import evaluate
import stouputils as stp
import nltk

# Download required NLTK data
required_nltk_resources: list[str] = ['punkt', 'punkt_tab']
for resource in required_nltk_resources:
    try:
        resource_path: str = nltk.data.find(f'tokenizers/{resource}')
        stp.info(f"NLTK {resource} already downloaded at {resource_path}")
    except LookupError:
        stp.info(f"Downloading NLTK {resource}...")
        nltk.download(resource)


def text_summarization(text: str, num_sentences: int = 5) -> str:
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
    ranked_sentences: list[tuple[int, float]] = sorted(
        ((i, score) for i, score in scores.items()),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top sentences
    selected_indices: list[int] = sorted([x[0] for x in ranked_sentences[:num_sentences]])
    
    # Combine sentences in original order
    summary: str = ' '.join([sentences[i] for i in selected_indices])
    
    return summary


@stp.measure_time()
@stp.handle_error()
def main():
    """
    {
        "rouge1": 0.2994305318391607,
        "rouge2": 0.11999540256446006,
        "rougeL": 0.2017001536256564,
        "rougeLsum": 0.2400318134730964
    }
    """
    dataset: pd.DataFrame = pd.read_csv("gemma10000.csv")

    # Generate summaries
    generated_texts: list[str] = [text_summarization(text) for text in dataset["original_text"]]

    # Evaluate the model
    stp.info(f"Evaluating the model on {len(dataset)} samples")
    rouge_score: evaluate.EvaluationModule = evaluate.load("rouge")
    eval_results: dict = rouge_score.compute(predictions=generated_texts, references=dataset["rewritten_text"])

    stp.info(stp.super_json_dump(eval_results))

if __name__ == "__main__":
    main()

