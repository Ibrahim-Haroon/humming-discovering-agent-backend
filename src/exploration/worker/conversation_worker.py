import os
import re
from typing import Optional
from datetime import datetime
from src.llm.llm_response_parser import LlmResponseParser
from src.llm.llm_template import CUSTOMER_ROLE, ANALYSIS_ROLE
from src.llm.llm_conversation_cache import LlmConversationCache
from src.core.model.conversation_node import ConversationNode
from src.core.model.conversation_edge import ConversationEdge
from src.core.model.conversation_graph import ConversationGraph
from src.core.enum.conversation_state import ConversationState
from src.exploration.worker.worker_context import WorkerContext
from src.llm.llm_prompt_contextualizer import LlmPromptContextualizer
from src.llm.service.llm_response_service import LlmResponseService
from src.rest.api.voice_api_client import VoiceApiClient, VoiceApiError
from src.speech.service.speech_transcribe_service import SpeechTranscribeService


class ConversationWorker:
    """Handles individual conversation paths during exploration"""

    def __init__(
            self,
            voice_client: VoiceApiClient,
            transcribe_service: SpeechTranscribeService,
            llm_service: LlmResponseService,
            graph: ConversationGraph
    ):
        self.__voice_client = voice_client
        self.__transcribe_service = transcribe_service
        self.__llm_service = llm_service
        self.__conversation_history = LlmConversationCache()
        self.__graph = graph

    def explore_path(self, context: WorkerContext) -> tuple[ConversationNode, ConversationEdge]:
        """
        Explores a conversation path by:
        1. Generating a customer prompt
        2. Making a call to get full conversation
        3. Analyzing transcript to determine state and next steps
        4. Creating appropriate nodes and edges in the conversation graph

        :param context: WorkerContext containing necessary information
        :returns Tuple of (new_node, edge_taken)
        :returns: Tuple of (new_node, edge_taken)
        :raises VoiceApiError: If voice API operations fail
        """
        recording_path: Optional[str] = None

        try:
            customer_prompt = context.metadata.get('prompt') or self.generate_new_prompt(context.current_node)

            raw_customer_speech = self.__llm_service.response(
                role=CUSTOMER_ROLE,
                prompt=customer_prompt
            )
            customer_speech = LlmResponseParser.parse_customer_prompt(raw_customer_speech)

            call_response = self.__voice_client.start_call(
                phone_number=context.phone_number,
                prompt=customer_speech
            )

            recording_path = self.__voice_client.get_recording(
                call_id=call_response.id
            )

            conversation_transcription = self.__transcribe_service.transcribe(recording_path)

            if not conversation_transcription.strip():
                raise ValueError("Received empty transcription from agent")

            # Update conversation history
            self.__conversation_history[context.business_type] = (
                f"PROMPT: {customer_speech}",
                f"CONVERSATION TRANSCRIPTION: {conversation_transcription}"
            )

            analysis_prompt = LlmPromptContextualizer.generate_analysis_prompt(
                context_type=context.business_type,
                customer_prompt=customer_speech,
                conversation_transcript=conversation_transcription
            )
            llm_raw_analysis = self.__llm_service.response(
                role=ANALYSIS_ROLE,
                prompt=analysis_prompt
            )

            llm_analysis = LlmResponseParser.parse(llm_raw_analysis)

            # Create new node
            new_node = ConversationNode(
                conversation_transcription=conversation_transcription,
                state=llm_analysis.state,
                parent_id=context.current_node.id,
                metadata={
                    'confidence': llm_analysis.confidence,
                    'reasoning': llm_analysis.reasoning,
                    'call_id': call_response.id,
                    'business_type': context.business_type,
                    'timestamp': datetime.now().isoformat()
                }
            )

            # Create edge representing this transition
            edge = ConversationEdge(
                source_node_id=context.current_node.id,
                target_node_id=new_node.id,
                response=llm_analysis.response,
                timestamp=datetime.now(),
                metadata={
                    'confidence': llm_analysis.confidence,
                    'call_id': call_response.id
                }
            )

            # Update graph
            self.__graph.add_node(new_node)
            self.__graph.add_edge(edge)

            return new_node, edge
        except VoiceApiError as e:
            # logging.error(f"Voice API error during exploration: {str(e)}")
            # Create error node to mark this path
            return self.__create_error_node(
                context.current_node.id,
                f"Voice API error: {str(e)}",
                e
            )

        except Exception as e:
            # logging.error(f"Exploration error: {str(e)}")
            return self.__create_error_node(
                context.current_node.id,
                f"Exploration error: {str(e)}",
                e
            )
        finally:
            # Cleanup temporary recording file
            if recording_path and os.path.exists(recording_path):
                try:
                    os.remove(recording_path)
                except NotImplementedError:
                    pass

    def __create_error_node(
            self,
            parent_id: str,
            error_message: str,
            exception: Exception
    ) -> tuple[ConversationNode, ConversationEdge]:
        """Creates an error node and edge for handling exploration failures"""
        error_node = ConversationNode(
            conversation_transcription="Error during exploration",
            state=ConversationState.ERROR,
            parent_id=parent_id,
            metadata={
                'error_message': error_message,
                'error_type': exception.__class__.__name__,
                'timestamp': datetime.now().isoformat()
            }
        )

        error_edge = ConversationEdge(
            source_node_id=parent_id,
            target_node_id=error_node.id,
            response="Error occurred",
            timestamp=datetime.now(),
            metadata={
                'error_message': error_message,
                'error_type': exception.__class__.__name__
            }
        )

        self.__graph.add_node(error_node)
        self.__graph.add_edge(error_edge)

        return error_node, error_edge

    def generate_new_prompt(self, node: ConversationNode) -> Optional[str]:
        """Generate unique conversation prompt for unexplored paths"""
        prompt = LlmPromptContextualizer.generate_customer_prompt(
            context_type=node.metadata.get('business_type'),
            conversation_history=self.__conversation_history[node.metadata.get('business_type')] or [],
            explored_paths=node.explored_responses
        )

        llm_response = self.__llm_service.response(
            role=CUSTOMER_ROLE,
            prompt=prompt
        )

        return LlmResponseParser.parse_customer_prompt(llm_response)

    def cleanup(self, context: WorkerContext) -> None:
        """Deletes conversation history when done with a path"""
        # Clear conversation history when done with this path
        if context.business_type in self.__conversation_history:
            del self.__conversation_history[context.business_type]
