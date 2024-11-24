import re


class TextNormalizer:
    """Handles text normalization operations for conversation analysis"""

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalizes text for comparison by removing extra whitespace, punctuation,
        and converting to lowercase.

        :param text: Raw text to normalize
        :return Normalized text string
        """
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.lower().split())
        return text
