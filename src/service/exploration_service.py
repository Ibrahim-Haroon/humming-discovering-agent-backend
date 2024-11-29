import time
import logging
from src.rest.api.voice_api_client import VoiceApiClient
from src.llm.service.llm_response_service import LlmResponseService
from src.core.model.conversation_graph import ConversationGraph
from src.exploration.conversation_explorer import ConversationExplorer
from src.exploration.worker.conversation_worker import ConversationWorker
from src.speech.service.speech_transcribe_service import SpeechTranscribeService


class ExplorationService:
    """
    Service for managing conversation exploration processes.
    """
    logger = logging.getLogger(__name__)

    def __init__(
            self,
            phone_number: str,
            business_type: str,
            llm_service: LlmResponseService,
            transcribe_service: SpeechTranscribeService,
            num_workers: int = 3,
            max_depth: int = 5
    ):
        self.phone_number = phone_number
        self.business_type = business_type
        self.llm_service = llm_service
        self.transcribe_service = transcribe_service
        self.num_workers = num_workers
        self.max_depth = max_depth
        self.explorer = None

    def __initialize_services(self):
        """Initialize all required services and workers"""
        voice_client = VoiceApiClient()
        graph = ConversationGraph()

        workers = [
            ConversationWorker(
                voice_client=voice_client,
                transcribe_service=self.transcribe_service,
                llm_service=self.llm_service,
                graph=graph
            )
            for _ in range(self.num_workers)
        ]

        self.explorer = ConversationExplorer(
            workers=workers,
            phone_number=self.phone_number,
            business_type=self.business_type,
            max_depth=self.max_depth,
            max_parallel=self.num_workers
        )

    def run_exploration(self):
        """Execute the exploration process"""
        self.__initialize_services()

        ExplorationService.logger.debug("Starting conversation exploration...")
        start_time = time.time()

        try:
            graph = self.explorer.explore()
            self.__log_results(graph, start_time)
            return graph

        except KeyboardInterrupt:
            ExplorationService.logger.debug("\nExploration stopped by user")
            self.stop()
        except Exception as e:
            ExplorationService.logger.debug(f"\nError during exploration: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the exploration process"""
        if self.explorer:
            self.explorer.stop()

    @staticmethod
    def __log_results(graph, start_time):
        """Log exploration results and statistics"""
        duration = time.time() - start_time
        ExplorationService.logger.debug("\nExploration completed!")
        ExplorationService.logger.debug(f"Total duration: {duration:.1f} seconds")
        ExplorationService.logger.debug("\nFinal Graph Statistics:")
        ExplorationService.logger.debug(f"Total nodes: {len(graph.nodes)}")
        ExplorationService.logger.debug(f"Total edges: {len(graph.edges)}")

        ExplorationService.logger.debug("\nDiscovered Terminal Paths:")
        for node in graph.nodes.values():
            if node.is_terminal():
                path = graph.get_path_to_node(node.id)
                ExplorationService.logger.debug("\nPath:")
                for edge in path:
                    ExplorationService.logger.debug(f"→ {edge.response}")
                ExplorationService.logger.debug(f"[{node.state.name}] {node.conversation_transcription[:200]}...")
