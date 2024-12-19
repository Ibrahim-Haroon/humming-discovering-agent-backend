import re
import logging
from uuid import UUID, uuid4
from typing import Optional
from src.graph.edge import Edge
from src.graph.node import Node
from src.graph.conversation_graph import ConversationGraph
from src.llm.models.llm_message import LlmMessage
from src.llm.models.llm_conversation_analysis import LlmConversationAnalysis
from src.llm.service.llm_response_service import LlmResponseService
from src.llm.template.llm_template import CUSTOMER_ROLE, ANALYSIS_ROLE
from src.llm.template.llm_template import LlmTemplate
from src.rest.api.hamming_voice_api_client import HammingVoiceApiClient
from src.rest.dto.hamming_call_response_dto import HammingCallResponseDTO
from src.speech.service.speech_transcribe_service import SpeechTranscribeService
from src.util.logging_config import setup_logging


class DiscoveryService:
    def __init__(
            self,
            business_type: str,
            business_number: str,
            hamming_api_client: HammingVoiceApiClient,
            llm_service: LlmResponseService,
            transcription_service: SpeechTranscribeService,
            conversation_graph: ConversationGraph,
            max_depth: Optional[int] = None
    ):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DiscoveryService:")
        self.logger.info(f"Business Type: {business_type}")
        self.logger.debug(f"Business Number: {business_number}")
        self.logger.debug(f"Max Depth: {max_depth if max_depth is not None else 'unlimited'}")

        self.__business_type = business_type
        self.__business_number = business_number
        self.__hamming_api_client = hamming_api_client
        self.__llm_service = llm_service
        self.__transcription_service = transcription_service
        self.__graph: ConversationGraph = conversation_graph
        self.__max_depth = max_depth

    def discover(self):
        """
        Start the discovery process, only discovering the initial node and then calls the __explore_node method
        that will recursively explore the conversation tree
        :return: None
        """
        self.logger.info("Starting discovery process...")

        initial_prompt = LlmTemplate.initial_customer_prompt(
            self.__business_type
        )
        self.logger.debug(f"Generated initial prompt: {initial_prompt[:100]}...")

        self.logger.info("Making initial call to agent...")
        transcription = self.__make_call(initial_prompt)
        self.logger.debug(f"Received initial transcription: {transcription}")

        root_node = Node(
            id=uuid4(),
            decision_point=transcription,
            assistant_message=LlmMessage(role="assistant", content=transcription),
            is_initial=True
        )
        self.logger.debug(f"Created root node with ID: {root_node.id}")

        self.__graph.add_node(root_node)
        self.logger.debug("Added root node to graph")

        self.logger.info("Starting node exploration...")
        self.__explore_node(root_node)
        self.logger.info("Discovery process completed")

    def __explore_node(self, curr_node: Node):
        self.logger.info(f"Exploring node {curr_node.id} at depth {curr_node.depth}")

        if self.__max_depth and curr_node.depth >= self.__max_depth:
            self.logger.info(f"Reached maximum depth ({self.__max_depth}), stopping exploration")
            return

        self.logger.debug("Analyzing conversation state...")
        analysis = self.__analyze_conversation_state(curr_node.id, curr_node.decision_point)
        self.logger.debug(f"Analysis: {analysis}")

        if analysis.is_terminal:
            self.logger.info("Reached terminal response, backtracking...")
            terminal_node = Node(
                id=uuid4(),
                decision_point="TERMINAL",
                assistant_message=LlmMessage(role="assistant", content="TERMINAL"),
                parent_id=curr_node.id,
                depth=curr_node.depth + 1,
                is_terminal=True
            )
            self.__graph.add_node(terminal_node)
            edge = Edge(
                source_node_id=curr_node.id,
                target_node_id=terminal_node.id,
                user_message=LlmMessage(role="user", content="TERMINAL")
            )
            self.__graph.add_edge(edge)
            return

        self.logger.debug(f"Generated {len(analysis.possible_responses)} possible responses")
        for idx, response in enumerate(analysis.possible_responses, 1):
            self.logger.info(f"Processing response {idx}/{len(analysis.possible_responses)}")
            self.logger.debug(f"Response: {response}")

            prompt = self.__generate_response_prompt(
                curr_node.id,
                response
            )
            self.logger.debug(f"Generated response prompt: {prompt}")

            self.logger.info("Making call to agent...")
            transcription = self.__make_call(prompt)
            self.logger.debug(f"Received transcription: {transcription}")

            new_node = Node(
                id=uuid4(),
                decision_point=transcription,
                assistant_message=LlmMessage(role="assistant", content=transcription),
                parent_id=curr_node.id,
                depth=curr_node.depth + 1
            )
            self.logger.debug(f"Created new node with ID: {new_node.id}")

            node_id = self.__graph.add_node(new_node)
            self.logger.debug(f"Added/retrieved node ID: {node_id}")

            edge = Edge(
                source_node_id=curr_node.id,
                target_node_id=node_id,
                user_message=LlmMessage(role="user", content=response)
            )
            self.__graph.add_edge(edge)

            if node_id == new_node.id:
                self.logger.info("Node is new, continuing exploration")
                self.__explore_node(new_node)
            else:
                self.logger.info("Node already exists, skipping further exploration")

    def __analyze_conversation_state(self, node_id: UUID, agent_response: str) -> LlmConversationAnalysis:
        self.logger.info("Analyzing conversation state")
        normalized_response = agent_response.lower().strip()

        transfer_patterns = [
            r"transferring to an agent"
            r"transfer(?:ring|red)?\s+(?:you|your\s+call)"
        ]

        callback_patterns = [
            r"call\s*(?:you\s*)?back",
            r"return\s*(?:your\s*)?call",
            r"(?:will|can|shall)\s+call\s+(?:you\s+)?back"
            r"contact you",
        ]

        unavailable_patterns = [
            r"cannot help",
            r"can't help",
            r"unable to assist",
            r"not able to help",
            r"(?:cannot|can\'t|unable\s+to)\s+(?:help|assist)"
        ]

        closing_patterns = [
            r"(?:appointment|service) (?:is )?confirm(?:ed)?",
        ]

        terminal_patterns = (
                transfer_patterns +
                callback_patterns +
                unavailable_patterns +
                closing_patterns
        )

        for pattern in terminal_patterns:
            if re.search(pattern, normalized_response, re.IGNORECASE):
                self.logger.debug(f"Terminal pattern matched: {pattern}")
                return LlmConversationAnalysis(
                    is_terminal=True,
                    possible_responses=None
                )
        self.logger.debug("No terminal pattern matched, proceeding with LLM analysis")

        history = self.__graph.build_conversation_history(node_id)
        contextualized_prompt = LlmTemplate.transcription_analysis_prompt(
            self.__business_type,
            agent_response,
        )

        response = self.__llm_service.response(
            role=ANALYSIS_ROLE,
            prompt=contextualized_prompt,
            conversation_history=history
        )

        try:
            terminal, possible_responses = response.split("|")
            is_terminal = terminal.strip() == "True"
            responses = [] if is_terminal else possible_responses.strip().split(";")

            return LlmConversationAnalysis(
                is_terminal=is_terminal,
                possible_responses=responses
            )
        except ValueError:
            self.logger.error(f"Failed to parse LLM response: {response}")
            raise ValueError(f"Unexpected LLM analysis format: {response}")

    def __generate_response_prompt(
            self,
            node_id: UUID,
            response: str
    ) -> str:
        self.logger.info("Generating response prompt")
        self.logger.debug(f"For node: {node_id}")

        history = self.__graph.build_conversation_history(node_id)
        self.logger.debug(f"Built conversation history: {history}")

        contextualized_prompt = LlmTemplate.response_customer_prompt(
            self.__business_type,
            response
        )
        self.logger.debug(f"Generated contextualized prompt")

        return self.__llm_service.response(
            role=CUSTOMER_ROLE,
            prompt=contextualized_prompt,
            conversation_history=history
        )

    def __make_call(self, prompt: str) -> str:
        self.logger.info("Making call")

        self.logger.debug("Starting call...")
        call_response: HammingCallResponseDTO = self.__hamming_api_client.start_call(
            self.__business_number,
            prompt
        )
        self.logger.debug(f"Call started with ID: {call_response.id}")

        self.logger.debug("Getting recording...")
        recording_path = self.__hamming_api_client.get_recording(call_response.id)
        self.logger.debug(f"Recording saved at: {recording_path}")

        self.logger.debug("Transcribing recording...")
        transcription = self.__transcription_service.transcribe(recording_path)
        self.logger.debug(f"Transcription complete")

        if not transcription.strip():
            self.logger.error("Received empty transcription from agent")
            raise ValueError("Received empty transcription from agent")

        return transcription
