import re

class DataPrepService:
    """
    Service to clean up the citation notations in the original article.
    """
    @staticmethod
    def clean_article(article: str) -> str:
        """
        Cleans citation notations from the article text.

        This method:
        - Removes inline citation numbers (e.g., "text1" or "text2") that appear
          as numeric notations in the middle of sentences.
        - Normalizes multiple spaces into a single space.

        Args:
            article (str): The original article text to clean.

        Returns:
            str: Cleaned article text without citation notations.
        """
        # Regex Explanation:
        # - \d+(?=\s): Matches one or more digits (\d+) followed by a space (lookahead).
        # - \s{2,}: Matches two or more consecutive spaces for normalization.
        cleaned_article = re.sub(r'\d+(?=\s)', '', article)  # Remove citation notations.
        cleaned_article = re.sub(r'\s{2,}', ' ', cleaned_article)  # Normalize spaces.
        return cleaned_article.strip()
