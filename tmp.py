from src.utils import convert_pdf_to_markdown
from src.chunk_embed import chunk_markdown, EmbedData, save_embeddings, load_embeddings
from src.index import QdrantVDB
from src.retriever import Retriever
from src.rag_engine import RAG
import os


# Disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


print("Identifying document layout...")
path = "attention.pdf"
name = path.rsplit('.', 1)[0]

print(f"filename: {name}")

found = any(
   os.path.isfile(f) and f.startswith(f"embeddings_{name}" + '.')
   for f in os.listdir('.')
)

# print(f">>>>>>>>{convert_pdf_to_markdown("./docs/attention.pdf")}")

if not found:
   markdown_text = convert_pdf_to_markdown("./docs/attention.pdf")

   print("------FINISHING-------------")
   print("Generating embeddings...")
   chunks = chunk_markdown(markdown_text)

   embeddata = EmbedData(batch_size=8)
   embeddata.embed(chunks)
   save_embeddings(embeddata, f"embeddings_{name}.pkl")

print("Loading embeddings...")
embeddata = load_embeddings(f"embeddings_{name}.pkl")

print("Indexing the document...")
database = QdrantVDB(collection_name="MultiMod_collection", vector_dim=len(embeddata.embeddings[0]), batch_size=7)
database.create_collection()
database.ingest_data(embeddata)


retriever = Retriever(database, embeddata=embeddata)
rag = RAG(retriever)
# print(f">>>>{rag.generate_context("generate an exam question facsimile about self attention mechanism exam of university")}")

print("Prompting....")
prompt = """
   
generate an example of question related to economy exam in university
"""
response_text = rag.query(prompt, "medium")
print(f"-----------> {response_text}")
