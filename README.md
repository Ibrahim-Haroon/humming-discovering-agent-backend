# System Architecture

```mermaid
flowchart TD
    A[main.py] --> B[ConversationExplorer]
    B --> C[WorkerPool]
    C --> |Creates| D[ConversationWorker]
    
    D --> |1. Generate Prompt| E[LlmPromptContextualizer]
    E --> |Uses| F[LlmResponseService]
    
    D --> |2. Make Call| G[VoiceApiClient]
    G --> |Webhook| H[HammingWebhookServer]
    
    D --> |3. Transcribe| I[SpeechTranscribeService]
    
    D --> |4. Analyze| J[LlmResponseParser]
    
    D --> |5. Update Graph| K[ConversationGraph]
    K --> |Contains| L[ConversationNode]
    K --> |Contains| M[ConversationEdge]
    
    L --> |Uses| N[NodeIdentifier]
    L --> |Uses| O[ResponseSimilarity]
    O --> |Uses| P[TextNormalizer]
    
    B --> |Tracks Progress| Q[ProgressTracker]
    Q --> |Updates| R[ExplorationStats]
```

# Get Running
Start both the frontend and backend services in both of respective directories using docker:
```
docker-compose up --build
```

Go to `localhost:5000` to view the graph being built in real-time. Just for simplicity the frontend polls the
backend every minute to get the latest nodes/edges, in a real-world scenario this would be done using websockets.