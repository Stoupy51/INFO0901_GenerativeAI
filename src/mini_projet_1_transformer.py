
# Imports
import pandas as pd
import evaluate
import stouputils as stp
from transformers import pipeline, BartTokenizer

# Initialize the model and tokenizer
@stp.measure_time(stp.progress)
def init_model() -> tuple[BartTokenizer, pipeline]:
    """ Initialize the BART model and tokenizer for summarization.
    
    Returns:
        tuple[BartTokenizer, pipeline]: Initialized tokenizer and summarization pipeline
    """
    model_name: str = "facebook/bart-large-cnn"
    tokenizer: BartTokenizer = BartTokenizer.from_pretrained(model_name)
    summarizer: pipeline = pipeline(
        "summarization",
        model=model_name,
        tokenizer=tokenizer,
        framework="pt"
    )
    return tokenizer, summarizer

def text_summarization(text: str, summarizer: pipeline, max_length: int = 130, min_length: int = 30) -> str:
    """ Summarize text using BART transformer model.
    
    Args:
        text              (str):        Input text to summarize
        summarizer        (pipeline):   Hugging Face summarization pipeline
        max_length        (int):        Maximum length of the summary
        min_length        (int):        Minimum length of the summary
        
    Returns:
        str: Generated summary of the input text
    """
    # Generate summary
    summary: list[dict] = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    
    return summary[0]['summary_text']

@stp.measure_time()
@stp.handle_error()
def main():
    """
    {
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
        "rougeLsum": 0
    }
    """
    dataset: pd.DataFrame = pd.read_csv("gemma10000.csv")

    # Initialize model
    stp.info("Initializing BART model and tokenizer...")
    _, summarizer = init_model()
    
    # Generate summaries
    stp.info(f"Generating summaries for {len(dataset)} texts...")
    generated_texts: list[str] = [
        text_summarization(text, summarizer) 
        for text in dataset["original_text"]
    ]

    # Evaluate the model
    stp.info(f"Evaluating the model on {len(dataset)} samples")
    rouge_score: evaluate.EvaluationModule = evaluate.load("rouge")
    eval_results: dict = rouge_score.compute(predictions=generated_texts, references=dataset["rewritten_text"])

    stp.info(stp.super_json_dump(eval_results))

if __name__ == "__main__":
    main()

