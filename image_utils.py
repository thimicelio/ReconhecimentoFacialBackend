import cv2


def preprocessar_imagem(caminho):
    img = cv2.imread(caminho)

    altura_max = 1000

    h, w = img.shape[:2]
    if h > altura_max:
        proporcao = altura_max / h
        img = cv2.resize(img, (int(w * proporcao), altura_max))

    return img
