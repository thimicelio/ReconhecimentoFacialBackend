import cv2
import numpy as np

IMAGEM_PESSOA = "treino6.jpeg"
IMAGEM_DOCUMENTO = "teste2.jpeg"

LIMIAR = 0.70


def carregar_imagem(caminho):
    imagem = cv2.imread(caminho)

    if imagem is None:
        print(f"Erro ao carregar: {caminho}")
        return None

    return imagem


def obter_rosto_unico(imagem, nome):
    classificador = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    variacoes = [
        ("0", imagem),
        ("90_cw", cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)),
        ("180", cv2.rotate(imagem, cv2.ROTATE_180)),
        ("90_ccw", cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    melhor_rosto = None
    melhor_area = 0
    melhor_rotacao = "0"

    for rotacao, imagem_rotacionada in variacoes:
        imagem_cinza = cv2.cvtColor(imagem_rotacionada, cv2.COLOR_BGR2GRAY)
        rostos = classificador.detectMultiScale(
            imagem_cinza,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for x, y, w, h in rostos:
            area = w * h
            if area > melhor_area:
                melhor_area = area
                melhor_rotacao = rotacao
                melhor_rosto = imagem_rotacionada[y:y + h, x:x + w]

    if melhor_rosto is None:
        print(f"Nenhum rosto encontrado em {nome}")
        return None

    print(f"Rosto selecionado em {nome}: rotação {melhor_rotacao}")
    return melhor_rosto


def processar_rosto(rosto):
    rosto_cinza = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
    rosto_cinza = cv2.resize(rosto_cinza, (200, 200))
    rosto_cinza = cv2.equalizeHist(rosto_cinza)
    return rosto_cinza


def obter_descritor_rosto(rosto_processado, nome):
    orb = cv2.ORB_create(nfeatures=500)
    _, descritor = orb.detectAndCompute(rosto_processado, None)

    if descritor is None or len(descritor) < 10:
        print(f"Não foi possível extrair características suficientes em {nome}")
        return None

    return descritor


def calcular_similaridade(descritor_a, descritor_b):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    correspondencias = bf.knnMatch(descritor_a, descritor_b, k=2)

    boas = []
    for dupla in correspondencias:
        if len(dupla) < 2:
            continue

        primeira, segunda = dupla
        if primeira.distance < 0.75 * segunda.distance:
            boas.append(primeira)

    base = max(len(correspondencias), 1)
    return len(boas) / base


def extrair_descritor_hog(imagem):
    grad_x = cv2.Sobel(imagem, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(imagem, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angulo = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)

    bins = (16 * angulo / (2 * np.pi)).astype(np.int32) % 16
    descritores = []
    tamanho_celula = 20

    for y in range(0, imagem.shape[0], tamanho_celula):
        for x in range(0, imagem.shape[1], tamanho_celula):
            bloco_bins = bins[y:y + tamanho_celula, x:x + tamanho_celula].ravel()
            bloco_mag = magnitude[y:y + tamanho_celula, x:x + tamanho_celula].ravel()
            hist = np.bincount(bloco_bins, weights=bloco_mag, minlength=16).astype(np.float32)
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


def validar():
    imagem_pessoa = carregar_imagem(IMAGEM_PESSOA)
    imagem_documento = carregar_imagem(IMAGEM_DOCUMENTO)

    if imagem_pessoa is None or imagem_documento is None:
        return

    rosto_pessoa = obter_rosto_unico(imagem_pessoa, "foto da pessoa")
    rosto_documento = obter_rosto_unico(imagem_documento, "documento")

    if rosto_pessoa is None or rosto_documento is None:
        return

    rosto_pessoa_processado = processar_rosto(rosto_pessoa)
    rosto_documento_processado = processar_rosto(rosto_documento)

    similaridade_hog = calcular_similaridade_hog(rosto_pessoa_processado, rosto_documento_processado)

    similaridade = similaridade_hog

    print(f"Similaridade HOG: {similaridade_hog:.4f}")
    print(f"Similaridade final: {similaridade:.4f}")

    if similaridade >= LIMIAR:
        print("✅ MESMA PESSOA")
    else:
        print("❌ PESSOA DIFERENTE")


if __name__ == "__main__":
    validar()