import os
import pandas as pd
import spacy
import textstat
import logging
from utils import load_excel, save_to_csv

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class AnalysisService:
    def __init__(self):
        self.nlp = nlp

    def calculate_readability(self, text: str) -> dict:
        """
        Calculate traditional readability metrics using TextStat.
        """
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "smog_index": textstat.smog_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
        }

    def calculate_scientific_metrics(self, text: str) -> dict:
        """
        Calculate scientific readability metrics.
        """
        doc = self.nlp(text)
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        total_words = len(text.split())
        complex_word_count = textstat.difficult_words(text)
        lexical_density = len([token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]) / total_words
        passive_sentences = sum(1 for sent in doc.sents if "by" in sent.text and "was" in sent.text)
        passive_voice_percentage = (passive_sentences / len(list(doc.sents))) * 100 if doc.sents else 0

        return {
            "avg_sentence_length": avg_sentence_length,
            "complex_word_percentage": (complex_word_count / total_words) * 100 if total_words else 0,
            "lexical_density": lexical_density,
            "passive_voice_percentage": passive_voice_percentage,
        }

def main():
    logging.info("Starting combined readability analysis...")

    # Define file paths
    data_dir = "data"
    excel_path = os.path.join(data_dir, "writing_polish_rcds.xlsx")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = os.path.join(results_dir, "readability_comparison_inExcel.csv")

    # Load Excel file
    logging.info(f"Loading Excel file: {excel_path}")
    df = load_excel(excel_path, sheet_name="Sheet1")

    # Ensure required columns exist
    if "Original" not in df.columns or "Polished" not in df.columns:
        raise ValueError("Excel file must contain 'Original' and 'Polished' columns.")

    # Initialize AnalysisService
    analysis_service = AnalysisService()

    # Storage for results
    readability_results = []

    # Process each article
    for idx, row in df.iterrows():
        article_id = idx + 1
        title = row.get("Title", "Unknown Title")
        year = row.get("Year", "Unknown Year")
        location = row.get("GRP", "Unknown").upper()

        # Analyze Original Text
        original_text = row["Original"]
        original_readability = analysis_service.calculate_readability(original_text)
        original_scientific = analysis_service.calculate_scientific_metrics(original_text)
        readability_results.append({
            "article_id": article_id,
            "title": title,
            "year": year,
            "location": location,
            "version": "original",
            **original_readability,
            **original_scientific
        })

        # Analyze Polished Text
        polished_text = row["Polished"]
        polished_readability = analysis_service.calculate_readability(polished_text)
        polished_scientific = analysis_service.calculate_scientific_metrics(polished_text)
        readability_results.append({
            "article_id": article_id,
            "title": title,
            "year": year,
            "location": location,
            "version": "excel_polished",
            **polished_readability,
            **polished_scientific
        })

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(readability_results)
    save_to_csv(results_df, results_csv_path)
    logging.info(f"Combined readability results saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
