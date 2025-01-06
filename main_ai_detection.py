import os
import logging
import pandas as pd
from utils import load_json, save_to_json, save_to_csv
from service.analysis_service import AnalysisService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def main():
    logging.info("Starting readability and AI detection analysis workflow...")

    # Define paths
    data_dir = "data"
    metadata_path = os.path.join(data_dir, "metadata.json")
    polished_articles_dir = "outputs/polished_articles"
    gptzero_output_base_dir = "outputs/gptzero_responses"
    os.makedirs(gptzero_output_base_dir, exist_ok=True)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # AI Detection configuration
    reps = ["original", "rep1", "rep2", "rep3"]  # Set the repetitions for which to perform AI detection

    # Load metadata
    metadata_records = load_json(metadata_path)
    logging.info("Metadata loaded successfully.")

    # Initialize AnalysisService
    analysis_service = AnalysisService()

    # Storage for readability and AI detection results
    readability_results = []
    ai_detection_results = []

    # Process all articles
    for article_id, metadata in metadata_records.items():
        logging.info(f"Processing article {article_id} for readability and AI detection...")

        # Readability result entry
        result_entry = {
            "article_id": article_id,
            "title": metadata["Title"],
            "year": metadata["Year"],
            "location": metadata["Location"],
            "original": None,
            "polished": {}
        }

        # Process each repetition, including original
        for rep in reps:
            # Define file paths for the text and response JSON
            if rep == "original":
                text_path = os.path.join(data_dir, f"article_{int(article_id):03}.txt")
                gptzero_output_dir = os.path.join(gptzero_output_base_dir, "original")
            else:
                text_path = os.path.join(polished_articles_dir, rep, f"output_{int(article_id):03}.txt")
                gptzero_output_dir = os.path.join(gptzero_output_base_dir, rep)

            os.makedirs(gptzero_output_dir, exist_ok=True)
            gptzero_save_path = os.path.join(gptzero_output_dir, f"ai_detection_{int(article_id):03}.json")

            # Skip if text file does not exist
            if not os.path.exists(text_path):
                logging.warning(f"Text file not found for article {article_id} in {rep}. Skipping...")
                continue

            # Read the text for readability analysis
            with open(text_path, "r", encoding="utf-8") as file:
                article_text = file.read()

            # Calculate letter length
            letter_length = len(article_text.replace(" ", ""))

            # Readability analysis
            readability_metrics = analysis_service.calculate_readability(article_text)
            readability_metrics.update(analysis_service.calculate_scientific_metrics(article_text))

            if rep == "original":
                result_entry["original"] = {**readability_metrics, "letter_length": letter_length}
            else:
                result_entry["polished"][rep] = {**readability_metrics, "letter_length": letter_length}

            # AI Detection: Skip API call if JSON exists
            if os.path.exists(gptzero_save_path):
                logging.info(f"GPTZero response exists for article {article_id} in {rep}. Loading JSON...")
                gptzero_response = load_json(gptzero_save_path)
            else:
                # Call GPTZero API and save response
                gptzero_response = analysis_service.detect_ai_text(article_text)
                save_to_json(gptzero_response, gptzero_save_path)
                logging.info(f"Saved GPTZero response for article {article_id} in {rep} to {gptzero_save_path}")

            # Extract relevant AI detection info
            document = gptzero_response.get("documents", [{}])[0]
            class_probabilities = document.get("class_probabilities", {})
            ai_detection_results.append({
                "article_id": article_id,
                "title": metadata.get("Title", "N/A"),
                "authors": "; ".join([" ".join([part for part in author.split() if "@" not in part]).strip() for author in metadata.get("Authors", ["N/A"])]),
                "year": metadata.get("Year", "N/A"),
                "location": metadata.get("Location", "N/A"),
                "version": rep,
                "letter_length": letter_length,
                "completely_generated_prob": round(document.get("completely_generated_prob", 0.0), 3),
                "human_prob": round(class_probabilities.get("human", 0.0), 3),
                "ai_prob": round(class_probabilities.get("ai", 0.0), 3),
                "mixed_prob": round(class_probabilities.get("mixed", 0.0), 3),
                "predicted_class": document.get("predicted_class", "N/A"),
                "confidence_category": document.get("confidence_category", "N/A"),
            })

        # Append readability results
        readability_results.append(result_entry)

    # Save readability results to JSON
    readability_json_path = os.path.join(results_dir, "readability_results.json")
    save_to_json(readability_results, readability_json_path)
    logging.info(f"Readability results saved to {readability_json_path}.")

    # Flatten and save readability results to CSV
    flat_readability_results = []
    for entry in readability_results:
        flat_readability_results.append({
            "article_id": entry["article_id"],
            "title": entry["title"],
            "year": entry["year"],
            "location": entry["location"],
            "version": "original",
            **entry["original"]
        })
        for rep, metrics in entry["polished"].items():
            flat_readability_results.append({
                "article_id": entry["article_id"],
                "title": entry["title"],
                "year": entry["year"],
                "location": entry["location"],
                "version": rep,
                **metrics
            })

    readability_csv_path = os.path.join(results_dir, "readability_comparison.csv")
    save_to_csv(pd.DataFrame(flat_readability_results), readability_csv_path)
    logging.info(f"Readability comparison saved to {readability_csv_path}.")

    # Save AI detection results to CSV with specified columns and version order
    ai_detection_results_ordered = []
    version_order = ["original", "rep1", "rep2", "rep3"]

    # Iterate over metadata to maintain additional attributes and version order
    for article_id, metadata in metadata_records.items():
        for version in version_order:
            relevant_results = [
                result for result in ai_detection_results
                if result["article_id"] == article_id and result["version"] == version
            ]
            if relevant_results:
                ai_detection_results_ordered.extend(relevant_results)

    # Filter and save the relevant fields
    filtered_results = [
        {
            "article_id": result["article_id"],
            "title": result.get("title", "N/A"),
            "authors": result.get("authors", "N/A"),
            "year": result.get("year", "N/A"),
            "location": result.get("location", "N/A"),
            "version": result.get("version", "N/A"),
            "letter_length": result.get("letter_length", 0),
            "human_prob": result.get("human_prob", 0.0),
            "ai_prob": result.get("ai_prob", 0.0),
            "mixed_prob": result.get("mixed_prob", 0.0),
            "completely_generated_prob": result.get("completely_generated_prob", 0.0),
            "predicted_class": result.get("predicted_class", "N/A"),
            "confidence_category": result.get("confidence_category", "N/A"),
        }
        for result in ai_detection_results_ordered
    ]

    ai_detection_csv_path = os.path.join(results_dir, "ai_detection_results_new.csv")
    save_to_csv(pd.DataFrame(filtered_results), ai_detection_csv_path)
    logging.info(f"AI detection results saved to {ai_detection_csv_path}.")

if __name__ == "__main__":
    main()
