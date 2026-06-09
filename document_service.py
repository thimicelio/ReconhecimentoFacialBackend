import os

import cv2
from paddleocr import PaddleOCR

from text_processing import (
    eh_label_nome,
    extrair_cpf,
    extrair_datas_por_contexto,
    extrair_nome_por_contexto,
    extrair_rg,
    limpar_nome_candidato,
    parece_nome_pessoa,
)


def configurar_ambiente_paddle():
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"


configurar_ambiente_paddle()

ocr = PaddleOCR(
    lang="pt",
    enable_mkldnn=False,
)


def preprocessar_imagem(caminho):
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


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


def extrair_texto(imagem):
    resultado = ocr.ocr(imagem, cls=False)
    if not resultado or not resultado[0]:
        return ""

    elementos = []
    for item in resultado[0]:
        box = item[0]
        texto = item[1][0]

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        elementos.append({"texto": texto, "x": min(xs), "y": min(ys)})

    elementos.sort(key=lambda e: e["y"])
    linhas = agrupar_por_linha(elementos)
    return "\n".join(linhas)


def extrair_dados(texto):
    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    dados = {"datas": {}}

    cpf_extraido = extrair_cpf(texto, linhas)
    if cpf_extraido:
        dados["cpf"] = cpf_extraido

    # Passar CPF extraído para evitar confusão com RG.
    rg_extraido = extrair_rg(texto, linhas, cpf_extraido=cpf_extraido)
    if rg_extraido:
        dados["rg"] = rg_extraido

    nome_contexto = extrair_nome_por_contexto(linhas)
    candidatos_nome = []

    for i, linha in enumerate(linhas):
        if eh_label_nome(linha):
            possiveis = []
            if i > 0:
                possiveis.append((linhas[i - 1], 1.0))
            if i + 1 < len(linhas):
                possiveis.append((linhas[i + 1], 0.8))

            for candidato, peso in possiveis:
                candidato_limpo = limpar_nome_candidato(candidato)
                if parece_nome_pessoa(candidato_limpo):
                    candidatos_nome.append((candidato_limpo, peso))
        else:
            linha_limpa = limpar_nome_candidato(linha)
            if parece_nome_pessoa(linha_limpa):
                candidatos_nome.append((linha_limpa, 0.5))

    if candidatos_nome:
        candidatos_nome.sort(key=lambda c: c[1], reverse=True)
        nome_heuristico = candidatos_nome[0][0]
        dados["nome"] = nome_contexto if nome_contexto else nome_heuristico
    elif nome_contexto:
        dados["nome"] = nome_contexto

    dados["datas"] = extrair_datas_por_contexto(linhas)

    return dados


def processar_documento(caminho_imagem):
    img_processada = preprocessar_imagem(caminho_imagem)
    texto = extrair_texto(img_processada)

    dados = extrair_dados(texto)
    return dados
