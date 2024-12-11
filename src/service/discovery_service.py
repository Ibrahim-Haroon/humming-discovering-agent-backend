from src.graph.node import Node
from src.graph.conversation_graph import ConversationGraph
from src.llm.cache.llm_conversation_cache import LlmConversationCache
from src.llm.history.llm_message import LlmMessage
from src.llm.service.llm_response_service import LlmResponseService
from src.llm.template.llm_prompt_contextualizer import LlmPromptContextualizer
from src.rest.api.hamming_voice_api_client import HammingVoiceApiClient
from src.rest.dto.hamming_call_response_dto import HammingCallResponseDTO
from src.speech.service.speech_transcribe_service import SpeechTranscribeService


class DiscoveryService:
    def __init__(
            self,
            business_type: str,
            business_number: str,
            hamming_api_client: HammingVoiceApiClient,
            llm_service: LlmResponseService,
            transcription_service: SpeechTranscribeService
    ):
        self.__business_type = business_type
        self.__business_number = business_number
        self.__hamming_api_client = hamming_api_client
        self.__llm_service = llm_service
        self.__transcription_service = transcription_service
        self.__graph: ConversationGraph = ConversationGraph()
        self.__conversation_cache = LlmConversationCache()

    def discover(self):
        initial_prompt = LlmPromptContextualizer.generate_initial_prompt(
            self.__business_type
        )

        transcription = self.__make_call(initial_prompt)

        if not transcription.strip():
            raise ValueError("Received empty transcription from agent")

        root_node = Node(
            id="",  # TODO: ?
            decision_point=transcription,
            assistant_message=LlmMessage(role="assistant", content=transcription),
            is_initial=True
        )
        self.__graph.add_node(root_node)
        self.__explore_node(root_node)

    def __explore_node(self, curr_node: Node):
        pass

    def __make_call(self, prompt: str) -> str:
        """start call -> get recording -> transcribe"""
        call_response: HammingCallResponseDTO = self.__hamming_api_client.start_call(
            self.__business_number,
            prompt
        )
        recording_path = self.__hamming_api_client.get_recording(call_response.id)
        return self.__transcription_service.transcribe(recording_path)


