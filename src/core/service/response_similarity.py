from difflib import SequenceMatcher
from typing import Set, Optional
from .text_normalizer import TextNormalizer


class ResponseSimilarity:
    """Handles comparison and similarity checking between responses"""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        :param similarity_threshold: Threshold for considering responses similar (0-1)
        """
        self.__similarity_threshold = similarity_threshold
        self.__normalizer = TextNormalizer()

    def is_similar(self, response1: str, response2: str) -> bool:
        """
        Determines if two responses are similar enough to be considered the same.

        :param response1: First response to compare
        :param response2: Second response to compare
        :return True if responses are considered similar, False otherwise
        """
        norm1 = self.__normalizer.normalize(response1)
        norm2 = self.__normalizer.normalize(response2)

        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= self.__similarity_threshold

    def find_similar(self, new_response: str, explored_responses: Set[str]) -> Optional[str]:
        """
        Checks if a similar response exists in the set of explored responses.

        :param new_response: The new response to check
        :param explored_responses: Set of responses that have already been explored
        :return Similar response if found, None otherwise
        """
        for explored in explored_responses:
            if self.is_similar(new_response, explored):
                return explored
        return None
