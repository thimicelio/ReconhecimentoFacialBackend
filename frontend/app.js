const form = document.querySelector("#validation-form");
const result = document.querySelector("#result");
const submitButton = document.querySelector("#submit-button");

const endpoint = "http://127.0.0.1:5000/api/validate-document";

function showJson(data) {
  result.textContent = JSON.stringify(data, null, 2);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);

  submitButton.disabled = true;
  submitButton.textContent = "Enviando...";
  result.textContent = "Aguardando resposta da API...";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { error: await response.text() };

    showJson(payload);
  } catch (error) {
    showJson({
      error: "Nao foi possivel chamar a API.",
      detail: error.message,
    });
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Enviar";
  }
});
