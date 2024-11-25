from flask import Flask
from flask_cors import CORS
from src.rest.api.graph_api import register_graph_routes
from src.service.exploration_service import ExplorationService


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
        self.__app.run(host=self.__host, port=self.__port)


def main():
    server = ApplicationServer()
    server.run()

    exploration_service = ExplorationService(
        phone_number="+14153580761",  # AC company number
        business_type="Air Conditioning and Plumbing company"
    )
    exploration_service.run_exploration()


if __name__ == "__main__":
    main()
