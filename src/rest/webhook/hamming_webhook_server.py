import threading
from urllib.parse import urlparse
from flask import Flask, request, jsonify, json
from src.rest.webhook.webhook_callback import WebhookCallback

app = Flask(__name__)
callback = WebhookCallback()


@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        recording_available = data.get('recording_available', False)
        if recording_available is True:
            call_id = data.get('id')
            with callback.callback_lock:
                if call_id in callback.callbacks:
                    callback.callbacks[call_id].put(data)
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'waiting'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def start_webhook_server():
    threading.Thread(
        target=lambda: app.run(host='127.0.0.1', port=8080, debug=False),
        daemon=True
    ).start()