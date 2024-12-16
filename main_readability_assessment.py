import os
import logging
import pandas as pd
from utils import load_json, save_to_json, save_to_csv
from service.analysis_service import AnalysisService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    """
    Perform readability assessments on original and polished texts.
    """
    logging.info("Starting readability assessment workflow...")

    # Define paths
    data_dir = "data"  # Directory for original texts
    polished_dir = "outputs/polished_articles"  # Base directory for polished texts
    metadata_path = os.path.join(data_dir, "metadata.json")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Control which repetitions to process
    reps = ["original", "rep1", "rep2", "rep3"]  # Control variable: includes original and reps

    # Load metadata
    try:
        metadata_records = load_json(metadata_path)
        logging.info("Metadata loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return

    # Initialize AnalysisService
    analysis_service = AnalysisService()

    # Storage for readability results
    readability_results = []

    # Process each article for original and polished repetitions
    for article_id, metadata in metadata_records.items():
        logging.info(f"Processing article {article_id}...")

        for rep in reps:
            # Set file path depending on rep (original vs polished)
            if rep == "original":
                text_path = os.path.join(data_dir, f"article_{int(article_id):03}.txt")
            else:
                text_path = os.path.join(polished_dir, rep, f"output_{int(article_id):03}.txt")

            if not os.path.exists(text_path):
                logging.warning(f"Text file not found for article {article_id} in {rep}. Skipping...")
                continue

            try:
                # Read the text
                with open(text_path, "r", encoding="utf-8") as file:
                    article_text = file.read()

                # Analyze readability
                readability_metrics = analysis_service.calculate_readability(article_text)
                scientific_metrics = analysis_service.calculate_scientific_metrics(article_text)

                # Combine results
                result_entry = {
                    "article_id": article_id,
                    "title": metadata.get("Title", "N/A"),
                    "year": metadata.get("Year", "N/A"),
                    "location": metadata.get("Location", "N/A"),
                    "version": rep,
                    **readability_metrics,
                    **scientific_metrics,
                }

                readability_results.append(result_entry)

            except Exception as e:
                logging.error(f"Error processing article {article_id} in {rep}: {e}")
                continue

    # Save results to CSV
    readability_csv_path = os.path.join(results_dir, "readability_results.csv")
    save_to_csv(pd.DataFrame(readability_results), readability_csv_path)
    logging.info(f"Readability results saved to {readability_csv_path}.")

    logging.info("Readability assessment workflow completed successfully.")


if __name__ == "__main__":
    main()
