from typing import override
from deepgram import Deepgram
from src.util.env import Env
from src.speech.service.speech_transcribe_service import SpeechTranscribeService


class DeepgramTranscribeService(SpeechTranscribeService):
    def __init__(self):
        self.__api_key = Env()["DEEPGRAM_API_KEY"]
        self.__dg = Deepgram(self.__api_key)

    @override
    def transcribe(self, audio_file_path: str) -> str:
        mime_type = 'audio/wav'

        options = {
            'punctuate': False,
            'model': 'general',
            'tier': 'enhanced'
        }

        with open(audio_file_path, 'rb') as f:
            _audio_ = {"buffer": f, "mimetype": mime_type}
            response = self.__dg.transcription.sync_prerecorded(_audio_, options)

        try:
            return response['results']['channels'][0]['alternatives'][0]['transcript']
        except KeyError:
            raise ValueError("Deepgram API response does not contain expected transcription")
