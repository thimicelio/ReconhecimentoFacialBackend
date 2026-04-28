from flask import Flask, jsonify, request
import os

app = Flask(__name__)

@app.route('/api/images', methods=['POST'])
def images():
    imagem1 = request.files.get("image1")
    imagem2 = request.files.get("image2")
    imagem1.save(f"./{imagem1.filename}")
    imagem2.save(f"./{imagem2.filename}")
    return jsonify({"status": "aaa"})

app.run()