import cv2
import numpy as np

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

def preprocessar_imagem(caminho):
    img = cv2.imread(caminho)

    altura_max = 1000

    h, w = img.shape[:2]
    if h > altura_max:
        proporcao = altura_max / h
        img = cv2.resize(img, (int(w * proporcao), altura_max))

    return img

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='pt',
    enable_mkldnn=False
)
def extrair_texto(imagem):
    resultado = ocr.predict(imagem)

    for pagina in resultado:
        textos = pagina["rec_texts"]
        boxes = pagina["rec_boxes"]

        elementos = []
        for texto, box in zip(textos, boxes):
            x1, y1, x2, y2 = box
            elementos.append({"texto": texto, "x": x1, "y": y1})

        elementos.sort(key=lambda e: e["y"])

        linhas = agrupar_por_linha(elementos)

        return "\n".join(linhas)
    
def agrupar_por_linha(elementos, tolerancia=15):
    linhas = []

    for el in elementos:
        encontrou = False

        for linha in linhas:
            if abs(linha[0]["y"] - el["y"]) < tolerancia:
                linha.append(el)
                encontrou = True
                break

        if not encontrou:
            linhas.append([el])

    linhas_ordenadas = []
    for linha in linhas:
        linha.sort(key=lambda e: e["x"])
        texto_linha = " ".join([e["texto"] for e in linha])
        linhas_ordenadas.append(texto_linha)

    return linhas_ordenadas

import re

def extrair_dados(texto):
    dados = {}

    linhas = [l.strip() for l in texto.split("\n") if l.strip()]

    cpf = re.search(r'\d{3}\.\d{3}\.\d{3}-\d{2}', texto)
    if cpf:
        dados["cpf"] = cpf.group()

    rg = re.search(r'\b[A-Z]{2}\d{7,}\b', texto)  # MG1234567
    if rg:
        dados["rg"] = rg.group()

    for i, linha in enumerate(linhas):
        if "NOME" in linha.upper() and i + 1 < len(linhas):
            dados["nome"] = linhas[i + 1]
            break

    if "nome" not in dados:
        for linha in linhas:
            if linha.isupper() and len(linha.split()) >= 2 and not any(char.isdigit() for char in linha):
                dados["nome"] = linha
                break

    datas = {}

    for i, linha in enumerate(linhas):
        datas_encontradas = re.findall(r'\d{2}/\d{2}/\d{4}', linha)

        if not datas_encontradas:
            continue

        contexto = linha.upper()

        for data in datas_encontradas:

            contexto_total = contexto
            if i > 0:
                contexto_total += " " + linhas[i-1].upper()

            if "NASC" in contexto_total:
                datas["nascimento"] = data

            elif "VALID" in contexto_total:
                datas["validade"] = data

            elif "EMISS" in contexto_total:
                datas["emissao"] = data

            else:
                if "nascimento" not in datas:
                    datas["nascimento"] = data
                elif "validade" not in datas:
                    datas["validade"] = data

    dados["datas"] = datas

    return dados

def processar_documento(caminho_imagem):
    img_processada = preprocessar_imagem(caminho_imagem)

    texto = extrair_texto(img_processada)

    print(texto)

    dados = extrair_dados(texto)

    return dados


dados = processar_documento("aluno1.jpg")

print("\n---------------")
print(dados)