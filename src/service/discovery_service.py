from uuid import UUID, uuid4
from typing import Optional
from src.graph.edge import Edge
from src.graph.node import Node
from src.graph.conversation_graph import ConversationGraph
from src.llm.history.llm_message import LlmMessage
from src.llm.models.llm_conversation_analysis import LlmConversationAnalysis
from src.llm.service.llm_response_service import LlmResponseService
from src.llm.template.llm_template import CUSTOMER_ROLE, ANALYSIS_ROLE
from src.llm.template.llm_template import LlmTemplate
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
            transcription_service: SpeechTranscribeService,
            max_depth: Optional[int] = None
    ):
        self.__business_type = business_type
        self.__business_number = business_number
        self.__max_depth = max_depth
        self.__hamming_api_client = hamming_api_client
        self.__llm_service = llm_service
        self.__transcription_service = transcription_service
        self.__graph: ConversationGraph = ConversationGraph()

    def discover(self):
        initial_prompt = LlmTemplate.initial_customer_prompt(
            self.__business_type
        )

        transcription = self.__make_call(initial_prompt)

        if not transcription.strip():
            raise ValueError("Received empty transcription from agent")

        root_node = Node(
            id=uuid4(),
            decision_point=transcription,
            assistant_message=LlmMessage(role="assistant", content=transcription),
            is_initial=True
        )
        self.__graph.add_node(root_node)
        self.__explore_node(root_node)

    def __explore_node(self, curr_node: Node):
        if self.__max_depth and curr_node.depth >= self.__max_depth:
            return

        analysis = self.__analyze_conversation_state(curr_node.decision_point)

        if analysis.is_terminal:
            curr_node.is_terminal = True
            return

        for response in analysis.possible_responses:
            prompt = self.__generate_response_prompt(
                curr_node.id,
                response
            )

            transcription = self.__make_call(prompt)
            new_node = Node(
                id=uuid4(),
                decision_point=transcription,
                assistant_message=LlmMessage(role="assistant", content=transcription),
                parent_id=curr_node.id,
                depth=curr_node.depth + 1
            )

            node_id = self.__graph.add_node(new_node)
            edge = Edge(
                source_node_id=curr_node.id,
                target_node_id=node_id,
                user_message=LlmMessage(role="user", content=response)
            )
            self.__graph.add_edge(edge)

            if node_id == new_node.id:
                self.__explore_node(new_node)

    def __analyze_conversation_state(self, agent_response: str) -> LlmConversationAnalysis:
        contextualized_prompt = LlmTemplate.transcription_analysis_prompt(
            self.__business_type,
            agent_response
        )

        response = self.__llm_service.response(
            role=ANALYSIS_ROLE,
            prompt=contextualized_prompt,
            conversation_history=None
        )

        try:
            terminal, possible_responses = response.split("|")
            is_terminal = terminal.strip() == "True"
            responses = [] if is_terminal else possible_responses.strip().split(";")

            return LlmConversationAnalysis(
                is_terminal=is_terminal,
                possible_responses=responses
            )
        except ValueError:  # expected format: bool|response1;response2;response3
            raise ValueError(f"Unexpected LLM analysis format: {response}")


    def __generate_response_prompt(
            self,
            node_id: UUID,
            response: str
    ) -> str:
        history = self.__graph.build_conversation_history(node_id)
        contextualized_prompt = LlmTemplate.response_customer_prompt(
            self.__business_type,
            response
        )

        return self.__llm_service.response(
            role=CUSTOMER_ROLE,
            prompt=contextualized_prompt,
            conversation_history=history
        )

    def __make_call(self, prompt: str) -> str:
        """start call -> get recording -> transcribe"""
        call_response: HammingCallResponseDTO = self.__hamming_api_client.start_call(
            self.__business_number,
            prompt
        )
        recording_path = self.__hamming_api_client.get_recording(call_response.id)
        return self.__transcription_service.transcribe(recording_path)
