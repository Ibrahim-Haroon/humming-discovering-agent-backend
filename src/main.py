import time
from src.rest.api.voice_api_client import VoiceApiClient
from src.core.model.conversation_graph import ConversationGraph
from src.exploration.conversation_explorer import ConversationExplorer
from src.exploration.worker.conversation_worker import ConversationWorker
from src.llm.service.openai_llm_response_service import OpenAILlmResponseService
from src.speech.service.deepgram_transcribe_service import DeepgramTranscribeService


def run_test_exploration():
    # Initialize services
    voice_client = VoiceApiClient()
    transcribe_service = DeepgramTranscribeService()
    llm_service = OpenAILlmResponseService()

    # Create shared graph
    graph = ConversationGraph()

    # Create workers
    workers = [
        ConversationWorker(
            voice_client=voice_client,
            transcribe_service=transcribe_service,
            llm_service=llm_service,
            graph=graph
        )
        for _ in range(3)  # Create 3 workers
    ]

    # Initialize explorer with AC company number
    explorer = ConversationExplorer(
        workers=workers,
        phone_number="+14153580761",  # AC company number
        business_type="Air Conditioning and Plumbing company",
        max_depth=2,  # Limit depth for testing
        max_parallel=3
    )

    print("Starting conversation exploration...")
    start_time = time.time()

    try:
        # Start exploration
        graph = explorer.explore()

        # Print progress updates every 5 seconds
        while not explorer.progress.is_complete():
            print("\n" + explorer.progress.get_progress_summary())
            time.sleep(5)

        # Final stats
        duration = time.time() - start_time
        print("\nExploration completed!")
        print(f"Total duration: {duration:.1f} seconds")
        print("\nFinal Graph Statistics:")
        print(f"Total nodes: {len(graph.nodes)}")
        print(f"Total edges: {len(graph.edges)}")

        # Print discovered paths
        print("\nDiscovered Terminal Paths:")
        for node in graph.nodes.values():
            if node.is_terminal():
                path = graph.get_path_to_node(node.id)
                print("\nPath:")
                for edge in path:
                    print(f"→ {edge.response}")
                print(f"[{node.state.name}] {node.agent_message}")

    except KeyboardInterrupt:
        print("\nExploration stopped by user")
        explorer.stop()
    except Exception as e:
        print(f"\nError during exploration: {str(e)}")
        explorer.stop()
    finally:
        # Cleanup
        for worker in workers:
            worker.cleanup()


if __name__ == "__main__":
    run_test_exploration()
