# rag_engine.py

from llama_index.llms.ollama import Ollama

class RAG:
    def __init__(self, retriever, llm_name="llama3.2"):
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.qa_prompt_tmpl_str = """Context information is below.
---------------------
{context}
---------------------

Given the context information above, answer the query in concise manner.
If you are not sure, respond with: 'I am unable to help you with that. Can I assist you with a different query?'

---------------------
Query: {query}
---------------------
Answer:"""

    def _setup_llm(self):
        return Ollama(model=self.llm_name)

    def generate_context(self, query):

        result = self.retriever.search(query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            context = entry["payload"]["context"]

            combined_prompt.append(context)

        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query):
        context = self.generate_context(query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
        response = self.llm.complete(prompt)
        return dict(response)['text']
