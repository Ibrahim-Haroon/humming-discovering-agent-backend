import time
import threading
from flask_cors import CORS
from flask import Flask, jsonify
from src.rest.api.voice_api_client import VoiceApiClient
from src.core.model.conversation_graph import ConversationGraph
from src.exploration.conversation_explorer import ConversationExplorer
from src.exploration.worker.conversation_worker import ConversationWorker
from src.llm.service.openai_llm_response_service import OpenAILlmResponseService
from src.speech.service.deepgram_transcribe_service import DeepgramTranscribeService

app = Flask(__name__)
CORS(app)


@app.route('/api/conversation-graph', methods=['GET'])
def get_graph():
    """
    Get the current conversation graph as JSON
    :return: JSON representation of the conversation graph
    ex. {
        "nodes": [
            {
                "id": "node1",
                "state": "INITIAL",
                "prompt": "Welcome to our AC and Plumbing company. How can I help you today?"
            },
            ...
        ],
        "edges": [
            {
                "source": "node1",
                "target": "node2"
            },
            ...
        ]
    }
    """
    graph = ConversationGraph()
    return jsonify({
        'nodes': [
            {
                'id': node.id,
                'state': node.state.name,
                'prompt': node.metadata.get('prompt', ''),
            }
            for node in graph.nodes.values()
        ],
        'edges': [
            {
                'source': edge.source_node_id,
                'target': edge.target_node_id
            }
            for edge in graph.edges
        ]
    })


def run_test_exploration():
    voice_client = VoiceApiClient()
    transcribe_service = DeepgramTranscribeService()
    llm_service = OpenAILlmResponseService()

    # Create shared graph
    graph = ConversationGraph()
    num_workers = 3

    # Create workers
    workers = [
        ConversationWorker(
            voice_client=voice_client,
            transcribe_service=transcribe_service,
            llm_service=llm_service,
            graph=graph
        )
        for _ in range(num_workers)  # Create 3 workers
    ]

    explorer = ConversationExplorer(
        workers=workers,
        phone_number="+14153580761",  # AC company number
        business_type="Air Conditioning and Plumbing company",
        max_depth=5,  # Limit depth for testing
        max_parallel=num_workers
    )

    print("Starting conversation exploration...")
    start_time = time.time()

    try:
        graph = explorer.explore()

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
                print(f"[{node.state.name}] {node.conversation_transcription[:200]}...")

    except KeyboardInterrupt:
        print("\nExploration stopped by user")
        explorer.stop()
    except Exception as e:
        print(f"\nError during exploration: {str(e)}")
        explorer.stop()


def run_flask():
    app.run(host='0.0.0.0', port=8000)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    run_test_exploration()
