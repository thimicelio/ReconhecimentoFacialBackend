import cv2


def preprocessar_imagem(caminho):
    img = cv2.imread(caminho)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.equalizeHist(img)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # cv2.imwrite("saida.png", img) # DEBUG ONLY

    return img

