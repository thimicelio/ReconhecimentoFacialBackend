from config_env import configurar_ambiente_paddle

configurar_ambiente_paddle()

from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="pt",
    enable_mkldnn=False,
)


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

        # Box com 4 pontos: usa menor x/y para ancorar e ordenar leitura.
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        elementos.append({"texto": texto, "x": min(xs), "y": min(ys)})

    elementos.sort(key=lambda e: e["y"])
    linhas = agrupar_por_linha(elementos)
    return "\n".join(linhas)
