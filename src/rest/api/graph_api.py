from flask import jsonify
from src.graph.conversation_graph import ConversationGraph


def register_graph_routes(app):
    """Register all graph-related routes with the Flask application"""

    @app.route('/api/conversation-graph', methods=['GET'])
    def get_graph():
        """
        Get the current conversation graph as a JSON object
        :return: JSON representation of the conversation graph
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
