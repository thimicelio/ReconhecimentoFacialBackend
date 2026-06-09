from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from document_service import processar_documento
from face_service import FaceComparisonError, comparar_faces

app = Flask("VexaAPI")


@app.after_request
def adicionar_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def _obter_threshold():
    valor = request.form.get("threshold", request.args.get("threshold"))
    if valor is None:
        raise ValueError("Parametro threshold e obrigatorio.")

    try:
        threshold = float(valor.replace(",", "."))
    except ValueError as exc:
        raise ValueError("Parametro threshold deve ser numerico.") from exc

    if threshold < 0 or threshold > 1:
        raise ValueError("Parametro threshold deve estar entre 0 e 1.")

    return threshold


def _salvar_upload(arquivo, pasta, nome_padrao):
    if arquivo is None or not arquivo.filename:
        raise ValueError(f"Arquivo {nome_padrao} e obrigatorio.")

    nome_seguro = secure_filename(arquivo.filename) or nome_padrao
    caminho = Path(pasta) / nome_seguro
    arquivo.save(caminho)
    return str(caminho)


@app.route("/api/images", methods=["POST", "OPTIONS"])
@app.route("/api/validate-document", methods=["POST", "OPTIONS"])
def validar_documento():
    if request.method == "OPTIONS":
        return "", 204

    try:
        threshold = _obter_threshold()

        with TemporaryDirectory() as pasta_temporaria:
            caminho_img_person = _salvar_upload(
                request.files.get("img_person"), pasta_temporaria, "img_person"
            )
            caminho_img_doc = _salvar_upload(
                request.files.get("img_doc"), pasta_temporaria, "img_doc"
            )

            comparacao = comparar_faces(caminho_img_person, caminho_img_doc, threshold)
            dados_documento = processar_documento(caminho_img_doc)

        return jsonify(
            {
                "same_person": comparacao["same_person"],
                "similarity": comparacao["similarity"],
                "threshold": comparacao["threshold"],
                "document": dados_documento,
            }
        )
    except (ValueError, FaceComparisonError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify(
            {"error": "Erro interno ao processar a requisicao.", "detail": str(exc)}
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
