from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import os
from openai import OpenAI
import faiss
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = []
index = None

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    global documents, index

    reader = PyPDF2.PdfReader(file.file)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]

    embeddings = [embed_text(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))
    documents = chunks

    return {"message": "PDF processado com sucesso."}


@app.post("/classificar")
async def classificar(
    altura: float = Form(...),
    area: float = Form(...),
    pavimentos: int = Form(...),
    ocupacao: str = Form(...)
):
    global documents, index

    query = f"""
    Classifique a edificação com:
    Altura: {altura}m
    Área: {area}m²
    Pavimentos: {pavimentos}
    Ocupação: {ocupacao}

    Informe:
    - Grupo de ocupação
    - Divisão
    - Classificação de altura
    - Classificação de área
    - Risco
    - Medidas obrigatórias
    - Necessidade de hidrantes
    - Necessidade de chuveiros automáticos
    - Fundamentação normativa
    """

    query_embedding = embed_text(query)
    D, I = index.search(np.array([query_embedding]), k=5)

    contexto = "\n".join([documents[i] for i in I[0]])

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é especialista em classificação de edificações conforme decretos e normas técnicas."},
            {"role": "user", "content": contexto + "\n\n" + query}
        ]
    )

    return {"resultado": resposta.choices[0].message.content}
