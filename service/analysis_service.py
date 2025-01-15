import spacy
import textstat
import requests
import os

class AnalysisService:
    """
    Service for analyzing readability metrics.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.gptzero_api_url = "https://api.gptzero.me/v2/predict/text"
        self.gptzero_api_key = os.getenv("GPTZERO_API_KEY")
        self.originality_api_url = "https://api.originality.ai/api/v2/scan"
        self.originality_api_key = os.getenv("ORIGINALITY_API_KEY")

    def detect_ai_text_gptzero(self, text: str) -> dict:
        """
        Detect AI-generated text using GPTZero API.
        """
        payload = {
            "document": text,
            "multilingual": False
        }
        headers = {
            "x-api-key": self.gptzero_api_key,
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.gptzero_api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"GPTZero API request failed: {e}")
            return {"error": str(e)}

    def detect_ai_text_originality(self, text: str) -> dict:
        """
        Detect AI-generated text using Originality.AI API v2.0.
        """
        if not text.strip():
            return {"error": "Content is empty or invalid."}

        url = self.originality_api_url
        payload = {
            "content": text.strip(),  # Ensure the content is clean
            "storeScan": False,
            "aiModel": "turbo",  # Use the appropriate AI model
            "scan_ai": True,
            "scan_plag": False,
            "scan_readability": False,
            "scan_grammar_spelling": False,
        }
        headers = {
            "Accept": "application/json",
            "X-OAI-API-KEY": self.originality_api_key,
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for HTTP issues
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"Originality.AI API request failed: {e}")
            print(f"Response content: {response.text}")
            return {"error": f"HTTP error: {e}"}
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return {"error": str(e)}

    def calculate_readability(self, text: str) -> dict:
        """
        Calculate traditional readability metrics for a given text.
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
        Calculate advanced metrics for professional clarity in scientific texts.
        """
        doc = self.nlp(text)
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        total_words = len(text.split())
        complex_words = textstat.difficult_words(text)
        lexical_density = len(
            [token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
        ) / total_words if total_words else 0
        passive_sentences = sum(
            1 for sent in doc.sents if "by" in sent.text and "was" in sent.text
        )
        passive_voice_percentage = (
            (passive_sentences / len(list(doc.sents))) * 100 if doc.sents else 0
        )

        return {
            "avg_sentence_length": avg_sentence_length,
            "complex_word_percentage": (complex_words / total_words) * 100 if total_words else 0,
            "lexical_density": lexical_density,
            "passive_voice_percentage": passive_voice_percentage,
        }

