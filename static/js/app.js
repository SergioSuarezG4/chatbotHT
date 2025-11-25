const form = document.getElementById("chatForm");
    const input = document.getElementById("userInput");
    const messages = document.getElementById("messages");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      messages.innerHTML += `<div class="msg user">Tú: ${text}</div>`;
      input.value = "";

      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: `message=${encodeURIComponent(text)}`
      });

      const data = await response.json();
      messages.innerHTML += `<div class="msg bot">Bot: ${data.response}</div>`;
      messages.scrollTop = messages.scrollHeight;
    });



    // Subir documento
const uploadForm = document.getElementById("uploadForm");
const uploadStatus = document.getElementById("uploadStatus");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = document.getElementById("fileInput").files[0];
  if (!file) {
    uploadStatus.innerHTML = "❌ Selecciona un archivo.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  uploadStatus.innerHTML = "📌 " + data.message;
});
