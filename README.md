# Multimodal-RAG
Multimodal RAG with Docling that lets you query PDFs containing text, tables, images, and formulas using a Retrieval-Augmented Generation pipeline. It leverages Docling for structured PDF parsing and Qdrant for fast vector search over embedded document chunks. Here's the demo: 

![Demo](images/MM-RAG.gif)



## What This Project Does
Link to Medium article: [Docling-Powered RAG: Querying Over Complex PDFs](https://medium.com/@pritigupta.ds/docling-powered-rag-querying-over-complex-pdfs-d99f5f58bc33)

This project is a **Streamlit-based application** on multimodal RAG that lets users:

* Upload a PDF document
* Extract structured **markdown using [Docling](https://github.com/microsoft/docling)**
* Replace embedded images with summaries
* Split the content into fixed length chunks
* Generate embeddings using `nomic-embed-text-v1.5`
* Store and index embeddings into a Qdrant vector database
* Use `Ollama + Llama 3.2` as the local LLM
* Enable chat-based querying through a RAG pipeline

All of this runs **locally**, with a clean UI and persistent chat history.

<img src="images/RAG-QueryType.png" width="100%">

## File Structure

```
docs/
|   ├── attention.pdf        # Test pdf for querying

src/
│   ├── chunk_embed.py       # Tokenization, chunking, and embedding
│   ├── index.py             # Qdrant Vector DB wrapper
│   ├── retriever.py         # Retriever class to fetch relevant chunks
│   ├── rag_engine.py        # RAG class combining retriever + LLM
│   └── utils.py             # Docling markdown + summary replacements

images/
│   ├── screenshot.png       # Interface screenshot
│   └── demo.mp4             # Video demo

output/
│   ├── output_attention.md            # Raw markdown output from Docling
│   ├── output_attention_textified.md # With images replaced by summaries

app.py                    # Main Streamlit app
README.md                 # You're reading it
```

---

## How It Works

1. **PDF Upload**: Users upload a PDF in the sidebar.
2. **Docling**: PDF is converted to markdown (with layout + tables + image data).
3. **Image Summaries**: Base64 images are replaced by prewritten summaries for clarity (obtained using OpenAI -4o for better results).
4. **Chunking + Embedding**:

   * Tokenized into 1024-token overlapping chunks.
   * Embedded using `nomic-embed-text-v1.5`.
5. **Indexing**: Embeddings are stored in a **Qdrant vector DB**.
6. **Querying**:

   * User queries are embedded.
   * Top relevant chunks are retrieved using **dot-product similarity**.
   * These are passed to **Ollama (Llama 3.2)** for final answer generation.



## Demo (Local Setup)

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant locally**:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Start Ollama with Llama 3.2**:

   ```bash
   ollama pull llama3
   ```

4. **Run the app**:

   ```bash
   streamlit run app.py
   ```


## References

* [Docling](https://github.com/docling-project/docling)
* [Streamlit Chat UI](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)
* [Qdrant Vector DB](https://qdrant.tech/)
* [Ollama + Llama 3](https://ollama.com/)
* [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)


## Acknowledgments

The UI of this project is adapted from the official [Streamlit Conversational App Tutorial](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps) and enhanced with **multimodal document understanding** using `Docling`, `Qdrant`, and `LlamaIndex`.


