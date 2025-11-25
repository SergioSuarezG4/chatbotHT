# ==========================================================
# main.py — Chatbot con GPT + RAG + FAISS (2025)
# ==========================================================

from flask import Flask, render_template, request, jsonify
import os
import random

# ==========================================================
# 🔧 Cargar variables de entorno ANTES de usar OpenAI
# ==========================================================
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

print("API KEY DETECTADA:", os.getenv("OPENAI_API_KEY"))

# ==========================================================
# 🔑 Cliente OpenAI (SDK oficial 2025)
# ==========================================================
from openai import OpenAI
client = OpenAI()


# Modelos previos (clusters del usuario)
from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_cluster

# Procesamiento de documentos
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_PATH = "vector_db"


# ==========================================================
# 🔧 Prueba automática de conexión con OpenAI
# ==========================================================
try:
    test = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hola, ¿funcionas?"}]
    )
    print("OpenAI funcionando →", test.choices[0].message.content)
except Exception as e:
    print("❌ Error al probar OpenAI:", e)


# ==========================================================
# 📄 Procesar y vectorizar documentos (RAG)
# ==========================================================
def procesar_documento(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        return None, f"❌ Tipo de archivo no soportado: {ext}"

    docs = loader.load()

    # Dividir contenido en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings y almacenamiento FAISS
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)

    vector_db.save_local(VECTOR_PATH)

    return True, "Documento procesado correctamente."


# ==========================================================
# 🤖 Modelo de Clusters del Usuario
# ==========================================================
app = Flask(__name__)
model, vectorizer = load_model()
if model is None:
    model, vectorizer = build_and_train_model(training_data, n_clusters=6)


RESPUESTAS = {
    0: ["¡Hola! 😊 ¿Cómo estás?", "¡Qué gusto saludarte!", "¿En qué puedo ayudarte hoy?"],
    1: ["Hasta luego 👋", "Nos vemos pronto.", "¡Cuídate! 😊"],
    2: ["Soy un asistente virtual creado para ayudarte 💻", "Pregúntame lo que quieras 😉"],
    3: ["¡Claro! ¿En qué puedo ayudarte?", "Cuéntame tu problema 🤖"],
    4: ["¡Gracias a ti! ❤️", "Me alegra ser de ayuda 😄"],
    5: ["Lamento eso 😔, puedo intentarlo nuevamente.", "Parece que algo no salió bien 😅"],
}


# ==========================================================
# 🌐 RUTAS FLASK
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# --- Subir documento ---
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"message": "❌ No enviaste archivo"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "❌ Nombre de archivo vacío"})

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    ok, msg = procesar_documento(path)
    return jsonify({"message": msg})


# --- CHAT ---
@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.form.get("message", "").strip()

    if not user_text:
        return jsonify({"response": "Por favor escribe algo 😅"})

    # ==========================================================
    # 1️⃣ RAG: Responder con GPT usando vector store
    # ==========================================================
    if os.path.exists(VECTOR_PATH):
        try:
            embeddings = OpenAIEmbeddings()
            vector_db = FAISS.load_local(
                VECTOR_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})

            docs = retriever.invoke(user_text)
            contexto = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
Eres un asistente experto. Responde usando ÚNICAMENTE el siguiente contexto
del documento. Si la información no está en el contexto, dilo claramente.

--- CONTEXTO ---
{contexto}
----------------

Pregunta del usuario:
{user_text}
"""

            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil y preciso."},
                    {"role": "user", "content": prompt}
                ]
            )

            return jsonify({"response": ai_response.choices[0].message.content})

        except Exception as e:
            print("⚠ Error en RAG:", e)

    # ==========================================================
    # 2️⃣ GPT normal si no hay documento
    # ==========================================================
    try:
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente amable y útil."},
                {"role": "user", "content": user_text}
            ]
        )
        return jsonify({"response": ai_response.choices[0].message.content})

    except Exception as e:
        print("⚠ Error con OpenAI:", e)

    # ==========================================================
    # 3️⃣ Backup: Modelo de clusters
    # ==========================================================
    cluster = predict_cluster(model, vectorizer, user_text)
    response = random.choice(
        RESPUESTAS.get(cluster, ["No estoy seguro de entender 😅, pero puedo intentarlo otra vez."])
    )
    return jsonify({"response": response})


# ==========================================================
# 🚀 Ejecutar servidor
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
