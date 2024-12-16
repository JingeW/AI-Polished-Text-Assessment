import os
import openai
import logging
import argparse
from dotenv import load_dotenv
from utils import load_excel, save_to_txt, save_to_json, load_json
from service.prompt_service import PromptService
from service.polish_service import PolishService
from service.data_prep_service import DataPrepService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def main(args):
    logging.info("Starting the polishing workflow...")

    # Log input parameters
    logging.info(f"Input Parameters: repetitions={args.repetitions}, model={args.model}, "
                 f"temperature={args.temperature}, prompt_version={args.prompt_version}, skip_data_prep={args.skip_data_prep}")

    # Load API key
    load_dotenv()
    api_key = os.getenv('API_KEY_1')

    if not api_key:
        raise ValueError("API key not found. Please set API_KEY_1 in your .env file.")
    logging.info("Loaded API key successfully.")

    # Initialize OpenAI client
    client = openai.Client(api_key=api_key)
    logging.info("OpenAI client initialized.")

    # Initialize PromptService
    prompt_service = PromptService()
    logging.info("PromptService initialized.")

    # Configurations for PolishService
    model = args.model
    temperature = args.temperature
    prompt_version = args.prompt_version
    repetitions = args.repetitions

    # Initialize PolishService
    polish_service = PolishService(
        client=client,
        prompt_service=prompt_service,
        prompt_version=prompt_version,
        model=model,
        temperature=temperature,
    )
    logging.info("PolishService initialized.")

    # Define paths
    data_dir = "data"
    excel_path = f"{data_dir}/writing_polish_rcds.xlsx"
    metadata_path = f"{data_dir}/metadata.json"
    output_dir = "outputs/polished_articles"
    os.makedirs(output_dir, exist_ok=True)

    # Load and clean articles if not skipping data prep
    if not args.skip_data_prep:
        logging.info("Data preparation step started...")
        os.makedirs(data_dir, exist_ok=True)

        logging.info(f"Loading Excel file: {excel_path}")
        df = load_excel(excel_path, sheet_name="Sheet1")
        logging.info(f"Loaded {len(df)} articles from the Excel file.")

        # Extract relevant columns
        articles = df["Original"].tolist()
        Titles = df["Title"].tolist()
        years = df["Year"].tolist()
        grp = df["GRP"].tolist()

        # Process metadata and save cleaned texts
        metadata_records = {}
        data_prep_service = DataPrepService()
        for idx, article in enumerate(articles):
            # Extract metadata
            Title = Titles[idx]
            year = years[idx]
            location = "USA" if grp[idx] == "USA" else "Asian"

            # Add metadata to the record
            metadata_records[idx + 1] = {
                "Title": Title,
                "Year": year,
                "Location": location,
            }

            cleaned_article = data_prep_service.clean_article(article)

            # Save cleaned text
            cleaned_file_name = f"article_{idx+1:03}.txt"
            cleaned_file_path = os.path.join(data_dir, cleaned_file_name)
            try:
                save_to_txt(cleaned_file_path, cleaned_article)
                logging.info(f"Cleaned text saved successfully to {cleaned_file_path}.")
            except Exception as e:
                logging.error(f"Error saving cleaned text to {cleaned_file_path}: {e}")

        # Save metadata to JSON
        try:
            save_to_json(metadata=metadata_records, file_path=metadata_path)
            logging.info(f"Metadata saved successfully to {metadata_path}.")
        except Exception as e:
            logging.error(f"Error saving metadata to {metadata_path}: {e}")
    else:
        # Skip data prep and load metadata
        logging.info("Skipping data preparation step. Loading pre-existing cleaned data...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run without --skip-data-prep first.")
        metadata_records = load_json(metadata_path)
        logging.info("Metadata loaded successfully.")

    # Perform repetitions for polished texts
    articles = [int(article_id) for article_id in metadata_records.keys()]
    for rep in range(1, repetitions + 1):
        logging.info(f"Starting repetition {rep}/{repetitions}...")
        # Create a subfolder for this repetition
        rep_folder = os.path.join(output_dir, f"rep{rep}")
        os.makedirs(rep_folder, exist_ok=True)

        # Process each article
        for idx, article_id in enumerate(articles):
            try:
                # Generate file name for polished text
                polished_file_name = f"output_{article_id:03}.txt"
                polished_file_path = os.path.join(rep_folder, polished_file_name)

                # Load cleaned article text
                cleaned_file_path = os.path.join(data_dir, f"article_{article_id:03}.txt")
                with open(cleaned_file_path, "r", encoding="utf-8") as file:
                    article_text = file.read()

                # Polish the article
                logging.info(f"Polishing article {article_id}/{len(articles)} in repetition {rep}...")
                polished_article = polish_service.polish_article(article_text)

                # Save polished text
                save_to_txt(polished_file_path, polished_article)
                logging.info(f"Polished article saved successfully to {polished_file_path}.\n")
            except Exception as e:
                logging.error(f"Error processing article {article_id} in repetition {rep}: {e}")

    logging.info("Workflow completed.")
    logging.info(f"Polished texts saved to repetitions under: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polish articles using OpenAI API.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions for polishing")
    parser.add_argument("--model", type=str, default="chatgpt-4o-latest", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the OpenAI API")
    parser.add_argument("--prompt_version", type=str, default="v1", help="Prompt version to use")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip the data preparation step if already done")
    args = parser.parse_args()

    main(args)
