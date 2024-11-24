import uuid
from datetime import datetime

from src.llm.llm_response_parser import LlmResponseParser
from src.llm.llm_template import ROLE
from src.llm.llm_conversation_cache import LlmConversationCache
from src.core.model.conversation_node import ConversationNode
from src.core.model.conversation_edge import ConversationEdge
from src.core.model.conversation_graph import ConversationGraph
from src.core.enum.conversation_state import ConversationState
from src.exploration.worker.worker_context import WorkerContext
from src.llm.llm_prompt_contextualizer import LlmPromptContextualizer
from src.speech.service.speech_transcribe_service import SpeechTranscribeService
from src.llm.service.llm_response_service import LlmResponseService
from src.rest.api.voice_api_client import VoiceApiClient


class ConversationWorker:
    """Handles individual conversation paths during exploration"""

    def __init__(
            self,
            voice_client: VoiceApiClient,
            transcribe_service: SpeechTranscribeService,
            llm_service: LlmResponseService,
            graph: ConversationGraph
    ):
        self.__id = uuid.uuid4()
        self.__voice_client = voice_client
        self.__transcribe_service = transcribe_service
        self.__llm_service = llm_service
        self.__conversation_history = LlmConversationCache()
        self.__graph = graph

    def explore_path(self, context: WorkerContext) -> tuple[ConversationNode, ConversationEdge]:
        """
        Explores a single conversation path by:
        1. Making a call
        2. Getting the agent's response
        3. Using LLM to generate a user response
        4. Creating the next node and edge

        :param context: WorkerContext containing necessary information
        :returns Tuple of (new_node, edge_taken)
        """
        call_id = self.__voice_client.start_call(context.phone_number)
        recording = self.__voice_client.get_recording(call_id)
        agent_message = self.__transcribe_service.transcribe(recording)

        history = self.__conversation_history[self.__id] or []

        llm_prompt = LlmPromptContextualizer.generate(
            context_type=context.business_type,
            current_agent_message=agent_message,
            conversation_history=history,
            explored_paths=context.current_node.explored_responses
        )
        llm_response = self.__llm_service.response(ROLE, llm_prompt)
        llm_response = LlmResponseParser.parse(llm_response)

        # Update conversation history
        self.__conversation_history[self.__id] = ("AGENT: " + agent_message, "USER: " + llm_response.response)

        # Create new node
        new_node = ConversationNode(
            agent_message=agent_message,
            state=llm_response.state,
            parent_id=context.current_node.id,
            metadata={
                'confidence': llm_response.confidence,
                'reasoning': llm_response.reasoning
            }
        )

        # Create edge representing this transition
        edge = ConversationEdge(
            source_node_id=context.current_node.id,
            target_node_id=new_node.id,
            response=llm_response.response,
            timestamp=datetime.now()
        )

        # Update graph
        self.__graph.add_node(new_node)
        self.__graph.add_edge(edge)

        return new_node, edge

    def cleanup(self) -> None:
        """Deletes conversation history when done with a path"""
        # Clear conversation history when done with this path
        if self.__id in self.__conversation_history:
            del self.__conversation_history[self.__id]

    @staticmethod
    def __determine_state(response: str) -> ConversationState:
        """
        Determines the conversation state based on the response

        :param response: LLM generated response
        :returns Appropriate ConversationState
        """
        # Check for terminal indicators
        if "[TERMINAL]" in response:
            if "agent" in response.lower() or "representative" in response.lower():
                return ConversationState.TERMINAL_TRANSFER
            return ConversationState.TERMINAL_SUCCESS

        return ConversationState.IN_PROGRESS
