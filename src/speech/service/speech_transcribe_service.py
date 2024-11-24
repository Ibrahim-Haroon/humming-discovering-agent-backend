from abc import ABC, abstractmethod


class SpeechTranscribeService(ABC):
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> str:
        """
        This method is used as a baseline behavior for all speech transcription providers

        :param audio_file_path: The path to the audio file to transcribe
        :return: transcription of the audio file
        :rtype: str
        """
        pass
