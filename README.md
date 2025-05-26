## Adaptive-RAG

**Adaptive-RAG** dynamically adjusts the amount of retrieved context based on query complexity, inspired by ["Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"](https://arxiv.org/pdf/2403.14403.pdf):  
- **Simple queries** retrieve fewer text chunks and are routed to the SLM running on the edge for fast, low-latency generation.  
- **Complex queries** retrieve more text chunks and are routed to the LLM in the cloud for deeper reasoning and more accurate responses.
