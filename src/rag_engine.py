import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from llama_index.llms.ollama import Ollama


load_dotenv(override=True)
LOCAL_SETTINGS = False

class RAG:
    def __init__(self, retriever):  #  llama3.2:latest or gpt-5 or tinyllama:1.1b 
        if LOCAL_SETTINGS:
            self.llm_name = 'llama3.2:latest'
        else:
            self.llm_name = 'gpt-5'

        self.llm = self._setup_llm()
        self.retriever = retriever

        # 🧠 Storico limitato a 3 messaggi (user, assistant, user)
        self.conversation_history = []

        # 📝 Memorizziamo separatamente l'ultima domanda dell'assistente
        self.last_question = None

        self.qa_prompt_tmpl_str = """
            Context information is below.
                <context/user_query>:
                ---------------------
                {context}
                ---------------------

                Based on the context information above, generate **one single open-ended question** related to the query below. Follow these rules precisely:

                1. **Number of questions:** 1  
                2. **Question type:** Open-ended (short essay or problem-solving)  
                3. **Content level:** University level (intermediate to advanced), aligned with the topic in `{context}`.  
                4. **LaTeX formatting:**  
                - Use inline math between single dollar signs `$...$`  
                - Use block math between double dollar signs `$$...$$`  
                - Do not escape backslashes
                5. **Tone:** Professional and educational
                6. **Output format:**  

                    ```
                    **Open-ended Question**

                    [your question here]
                    ```

                8. **Interaction flow:**  
                - Step 1: Generate and display the question only.  
                - Step 2: Wait for the user to provide their answer.  
                - Step 3: After receiving the user’s answer, evaluate it:
                    - Positive feedback if correct or partially correct, plus explanation.
                    - Constructive feedback and correct answer if incorrect.

                9. **Language:** English only.

                ---------------------
                Query: {query}
                ---------------------
                Answer:
        """

        self.evaluation_prompt = """
        Evaluate the following user answer based on the question asked previously.

        Question:
        {question}

        User Answer:
        {user_answer}

        Your task:
        - Check if the answer is relevant and correct regarding the question.
        - Provide constructive feedback.
        - If correct or partially correct → positive feedback + short explanation.
        - If incorrect → constructive feedback + correct answer.
        - Answer in English.
        """

    def _setup_llm(self):
        if LOCAL_SETTINGS:
            return Ollama(model=self.llm_name, request_timeout=300000)   # in mms
        else:
            return AzureOpenAI(
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            )

    def generate_context(self, query):
        result = self.retriever.search(query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            context_str = entry["payload"]["context"]
            combined_prompt.append(context_str)

        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query):
        """
        Questo metodo gestisce:
        - Prima interazione: genera una domanda aperta
        - Seconda interazione: valuta la risposta rispetto all'ultima domanda
        """
        # Se abbiamo già una domanda precedente, allora la nuova query è una risposta
        if self.last_question:
            evaluation_prompt = self.evaluation_prompt.format(
                question=self.last_question,
                user_answer=query
            )

            if LOCAL_SETTINGS:
                response = self.llm.complete(evaluation_prompt)
                # print(f">>>>> {response}")
                assistant_reply = dict(response)['text']       
            else:
                response = self.llm.chat.completions.create(
                    model=self.llm_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful evaluator."},
                        {"role": "user", "content": evaluation_prompt},
                    ]
                )
                assistant_reply = response.choices[0].message.content



            # Reset dello storico dopo la valutazione
            self.conversation_history = []
            self.last_question = None

            return assistant_reply

        # Altrimenti è la prima domanda → generiamo una open-ended question
        context = self.generate_context(query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)

        if LOCAL_SETTINGS:
            response = self.llm.complete(prompt)
            # print(f">>>>> {response}")
            assistant_reply = dict(response)['text']
        else:
            response = self.llm.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            assistant_reply = response.choices[0].message.content
        


        # Aggiorna storico (massimo 3 messaggi)
        self.conversation_history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_reply}
        ]

        # Salva l'ultima domanda dell'assistente per la valutazione futura
        self.last_question = assistant_reply

        return assistant_reply
    