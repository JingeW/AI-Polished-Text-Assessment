import os
import logging
import pandas as pd
from dotenv import load_dotenv
from utils import load_json, save_to_json
from service.analysis_service import AnalysisService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    load_dotenv() 
    logging.info("Starting AI detection analysis workflow with GPTZero and Originality.AI...")

    # Define paths
    data_dir = "data"
    metadata_path = os.path.join(data_dir, "metadata.json")
    polished_articles_dir = "outputs/polished_articles"
    gptzero_output_base_dir = "outputs/gptzero_responses"
    originality_output_base_dir = "outputs/originalityai_responses"
    os.makedirs(gptzero_output_base_dir, exist_ok=True)
    os.makedirs(originality_output_base_dir, exist_ok=True)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # AI Detection configuration
    reps = ["original", "rep1", "rep2", "rep3"]  # Set the repetitions for which to perform AI detection

    # Load metadata
    metadata_records = load_json(metadata_path)
    logging.info("Metadata loaded successfully.")

    # Initialize AnalysisService
    analysis_service = AnalysisService()

    # Storage for AI detection results
    gptzero_results_list = []
    originality_results_list = []

    # Process all articles
    for article_id, metadata in metadata_records.items():
        logging.info(f"Processing article {article_id} for AI detection...")

        # Process each repetition, including original
        for rep in reps:
            # Define file paths for the text and response JSON
            if rep == "original":
                text_path = os.path.join(data_dir, f"article_{int(article_id):03}.txt")
                gptzero_output_dir = os.path.join(gptzero_output_base_dir, "original")
                originality_output_dir = os.path.join(originality_output_base_dir, "original")
            else:
                text_path = os.path.join(polished_articles_dir, rep, f"output_{int(article_id):03}.txt")
                gptzero_output_dir = os.path.join(gptzero_output_base_dir, rep)
                originality_output_dir = os.path.join(originality_output_base_dir, rep)

            os.makedirs(gptzero_output_dir, exist_ok=True)
            os.makedirs(originality_output_dir, exist_ok=True)

            gptzero_save_path = os.path.join(gptzero_output_dir, f"ai_detection_{int(article_id):03}.json")
            originality_save_path = os.path.join(originality_output_dir, f"ai_detection_{int(article_id):03}.json")

            # Skip if text file does not exist
            if not os.path.exists(text_path):
                logging.warning(f"Text file not found for article {article_id} in {rep}. Skipping...")
                continue

            # Read the text for analysis
            with open(text_path, "r", encoding="utf-8") as file:
                article_text = file.read()

            # Calculate letter length
            letter_length = len(article_text.replace(" ", ""))

            # GPTZero Detection
            if not os.path.exists(gptzero_save_path):
                logging.info(f"Running GPTZero detection for article {article_id} in {rep}...")
                gptzero_response = analysis_service.detect_ai_text_gptzero(article_text)
                save_to_json(gptzero_response, gptzero_save_path)
                logging.info(f"Saved GPTZero response for article {article_id} in {rep} to {gptzero_save_path}")
            else:
                logging.info(f"GPTZero response exists for article {article_id} in {rep}. Skipping...")
                gptzero_response = load_json(gptzero_save_path)

            # Extract GPTZero relevant info
            document = gptzero_response.get("documents", [{}])[0]
            class_probabilities = document.get("class_probabilities", {})

            gptzero_results_list.append({
                "article_id": article_id,
                "title": metadata.get("Title", "N/A"),
                "authors": "; ".join([" ".join([part for part in author.split() if "@" not in part]).strip() for author in metadata.get("Authors", ["N/A"])]),
                "year": metadata.get("Year", "N/A"),
                "location": metadata.get("Location", "N/A"),
                "version": rep,
                "completely_generated_prob": round(document.get("completely_generated_prob", 0.0), 3),
                "human_prob": round(class_probabilities.get("human", 0.0), 3),
                "ai_prob": round(class_probabilities.get("ai", 0.0), 3),
                "predicted_class": document.get("predicted_class", "N/A"),
                "confidence_category": document.get("confidence_category", "N/A"),
                "letter_length": letter_length
            })

            # Originality.AI Detection
            if not os.path.exists(originality_save_path):
                logging.info(f"Running Originality.AI detection for article {article_id} in {rep}...")
                originality_response = analysis_service.detect_ai_text_originality(article_text)
                save_to_json(originality_response, originality_save_path)
                logging.info(f"Saved Originality.AI response for article {article_id} in {rep} to {originality_save_path}")
            else:
                logging.info(f"Originality.AI response exists for article {article_id} in {rep}. Skipping...")
                originality_response = load_json(originality_save_path)

            # Extract Originality.AI relevant info
            ai_classification = originality_response.get("ai", {}).get("classification", {})
            ai_confidence = originality_response.get("ai", {}).get("confidence", {})

            originality_results_list.append({
                "article_id": article_id,
                "title": metadata.get("Title", "N/A"),
                "authors": "; ".join([" ".join([part for part in author.split() if "@" not in part]).strip() for author in metadata.get("Authors", ["N/A"])]),
                "year": metadata.get("Year", "N/A"),
                "location": metadata.get("Location", "N/A"),
                "version": rep,
                "AI_classification": ai_classification.get("AI", "N/A"),
                "Original_classification": ai_classification.get("Original", "N/A"),
                "AI_confidence": ai_confidence.get("AI", "N/A"),
                "Original_confidence": ai_confidence.get("Original", "N/A"),
                "letter_length": letter_length
            })
    # Save GPTZero results to Excel
    gptzero_results_df = pd.DataFrame(gptzero_results_list)
    gptzero_excel_path = os.path.join(results_dir, "gptzero_results.xlsx")
    gptzero_results_df.to_excel(gptzero_excel_path, index=False)
    logging.info(f"GPTZero results saved to {gptzero_excel_path}.")

    # Save Originality.AI results to Excel
    originality_results_df = pd.DataFrame(originality_results_list)
    originality_excel_path = os.path.join(results_dir, "originality_ai_results.xlsx")
    originality_results_df.to_excel(originality_excel_path, index=False)
    logging.info(f"Originality.AI results saved to {originality_excel_path}.")

if __name__ == "__main__":
    main()
