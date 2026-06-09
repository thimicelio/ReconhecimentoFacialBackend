from field_extractors import (
    eh_label_nome,
    extrair_cpf,
    extrair_datas_por_contexto,
    extrair_nome_por_contexto,
    extrair_rg,
    limpar_nome_candidato,
    parece_nome_pessoa,
)
from image_utils import preprocessar_imagem
from ocr_reader import extrair_texto


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
