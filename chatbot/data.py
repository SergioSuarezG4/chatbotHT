# chatbot/data.py
# ==========================================================
# 💬 DATA PARA MODELO NO SUPERVISADO
# Agrupa frases de conversación cotidianas sin etiquetas.
# ==========================================================

training_data = [
    # 👋 Saludos
    "hola", "buenos días", "buenas tardes", "buenas noches",
    "qué tal", "cómo estás", "cómo te va", "qué más", "hey", "saludos",

    # 👋 Despedidas
    "adiós", "hasta luego", "nos vemos", "chao", "me despido",
    "hasta pronto", "cuídate", "que estés bien",

    # 🙋‍♂️ Preguntas personales
    "cómo te llamas", "cuál es tu nombre", "quién eres",
    "de dónde eres", "qué eres", "cuál es tu función",

    # ⚙️ Preguntas sobre capacidades
    "qué puedes hacer", "qué sabes hacer", "para qué sirves",
    "puedes ayudarme", "qué funciones tienes", "cuál es tu trabajo",

    # 🆘 Pedidos de ayuda o información
    "necesito ayuda", "ayúdame por favor", "tengo un problema",
    "no entiendo algo", "explícame esto", "cómo funciona esto",
    "dame información", "muéstrame un ejemplo",

    # ℹ️ Conversaciones informativas
    "qué hora es", "dónde estás", "cuál es la capital de colombia",
    "qué día es hoy", "qué clima hace", "cuál es la fecha de hoy",

    # ❤️ Expresiones de gratitud
    "gracias", "muchas gracias", "te agradezco",
    "muy amable", "gracias por tu ayuda",

    # 😠 Frustración o queja
    "no me sirves", "no entiendo nada", "esto no funciona",
    "no me ayudas", "eres inútil", "no sabes responder",
]
