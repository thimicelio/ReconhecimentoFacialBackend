import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


class FaceComparisonError(ValueError):
    pass


def obter_caminho_haarcascade():
    origem = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    destino_dir = Path(tempfile.gettempdir()) / "reconhecimento_facial_backend_cv2"
    destino = destino_dir / origem.name

    if not origem.exists():
        raise FaceComparisonError(f"Arquivo haarcascade nao encontrado: {origem}")

    if not destino.exists():
        destino_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origem, destino)

    return str(destino)


def carregar_imagem(caminho):
    imagem = cv2.imread(caminho)
    if imagem is None:
        raise FaceComparisonError(f"Nao foi possivel carregar a imagem: {caminho}")
    return imagem


def obter_rosto_unico(imagem, nome):
    classificador = cv2.CascadeClassifier(obter_caminho_haarcascade())
    if classificador.empty():
        raise FaceComparisonError("Nao foi possivel carregar o classificador de rosto.")

    variacoes = [
        ("0", imagem),
        ("90_cw", cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)),
        ("180", cv2.rotate(imagem, cv2.ROTATE_180)),
        ("90_ccw", cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    melhor_rosto = None
    melhor_area = 0

    for _, imagem_rotacionada in variacoes:
        imagem_cinza = cv2.cvtColor(imagem_rotacionada, cv2.COLOR_BGR2GRAY)
        rostos = classificador.detectMultiScale(
            imagem_cinza,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for x, y, w, h in rostos:
            area = w * h
            if area > melhor_area:
                melhor_area = area
                melhor_rosto = imagem_rotacionada[y : y + h, x : x + w]

    if melhor_rosto is None:
        raise FaceComparisonError(f"Nenhum rosto encontrado em {nome}")

    return melhor_rosto


def processar_rosto(rosto):
    rosto_cinza = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
    rosto_cinza = cv2.resize(rosto_cinza, (200, 200))
    rosto_cinza = cv2.equalizeHist(rosto_cinza)
    return rosto_cinza


def extrair_descritor_hog(imagem):
    grad_x = cv2.Sobel(imagem, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(imagem, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angulo = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)

    bins = (16 * angulo / (2 * np.pi)).astype(np.int32) % 16
    descritores = []
    tamanho_celula = 20

    for y in range(0, imagem.shape[0], tamanho_celula):
        for x in range(0, imagem.shape[1], tamanho_celula):
            bloco_bins = bins[y : y + tamanho_celula, x : x + tamanho_celula].ravel()
            bloco_mag = magnitude[y : y + tamanho_celula, x : x + tamanho_celula].ravel()
            hist = np.bincount(
                bloco_bins, weights=bloco_mag, minlength=16
            ).astype(np.float32)
            norma = np.linalg.norm(hist)
            if norma > 0:
                hist /= norma
            descritores.append(hist)

    return np.concatenate(descritores)


def calcular_similaridade_hog(imagem_a, imagem_b):
    desc_a = extrair_descritor_hog(imagem_a)
    desc_b = extrair_descritor_hog(imagem_b)

    norma = np.linalg.norm(desc_a) * np.linalg.norm(desc_b)
    if norma == 0:
        return 0.0

    return float(np.dot(desc_a, desc_b) / norma)


def comparar_faces(caminho_img_person, caminho_img_doc, threshold):
    imagem_pessoa = carregar_imagem(caminho_img_person)
    imagem_documento = carregar_imagem(caminho_img_doc)

    rosto_pessoa = obter_rosto_unico(imagem_pessoa, "img_person")
    rosto_documento = obter_rosto_unico(imagem_documento, "img_doc")

    rosto_pessoa_processado = processar_rosto(rosto_pessoa)
    rosto_documento_processado = processar_rosto(rosto_documento)

    similaridade = calcular_similaridade_hog(
        rosto_pessoa_processado, rosto_documento_processado
    )

    return {
        "same_person": similaridade >= threshold,
        "similarity": similaridade,
        "threshold": threshold,
    }
