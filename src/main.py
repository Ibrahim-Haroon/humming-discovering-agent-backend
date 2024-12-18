import logging
import threading
from flask import Flask
from flask_cors import CORS
from src.util.logging_config import setup_logging
from src.graph.conversation_graph import ConversationGraph
from src.rest.api.graph_api import register_graph_routes
from src.llm.service.openai_llm_response_service import OpenAILlmResponseService
from src.rest.api.hamming_voice_api_client import HammingVoiceApiClient
from src.service.discovery_service import DiscoveryService
from src.speech.service.deepgram_transcribe_service import DeepgramTranscribeService


class ApplicationServer:
    """
    Server configuration and initialization for the Flask application.
    """

    def __init__(self, host='0.0.0.0', port=8000):
        self.__app = Flask(__name__)
        self.__host = host
        self.__port = port
        self.__configure_app()

    def __configure_app(self):
        """Configure Flask application with middleware and routes"""
        CORS(self.__app)
        register_graph_routes(self.__app)

    def run(self):
        """Start the Flask server"""
        threading.Thread(
            target=lambda: self.__app.run(
                host=self.__host,
                port=self.__port,
                debug=False
            ),
            daemon=True
        ).start()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting application...")
    server = ApplicationServer()
    server.run()

    logger.info("Initializing discovery service...")
    discovery_service = DiscoveryService(
        business_type="Air Conditioning and Plumbing company",
        business_number="+14153580761",  # AC company number
        llm_service=OpenAILlmResponseService(),
        transcription_service=DeepgramTranscribeService(),
        hamming_api_client=HammingVoiceApiClient(),
        conversation_graph=ConversationGraph(node_similarity_threshold=0.9),
        max_depth=5
    )

    discovery_service.discover()


if __name__ == "__main__":
    main()
