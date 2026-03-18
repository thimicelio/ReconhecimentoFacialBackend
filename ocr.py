import pytesseract
import cv2
import os

# Caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Caminho dos idiomas
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Ler imagem
imagem = cv2.imread('CNH.jpeg')

# Pré-processamento
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(cinza, 150, 255, cv2.THRESH_BINARY)

# OCR com português
texto = pytesseract.image_to_string(thresh, lang='por')

print(texto)