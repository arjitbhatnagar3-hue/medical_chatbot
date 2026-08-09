import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate



load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task = "text-generation",
    temperature=0.2,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


CUSTOM_PROMPT = """
You are a medical assistant.

Answer the question ONLY using the provided context.
Do not use outside knowledge.
If the answer cannot be found in the context, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT,
    input_variables=["context", "question"]
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 2}
)



def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
)

    final_prompt = prompt.format(
        context=context,
        question=question
)

    response = model.invoke(final_prompt)

    return response.content