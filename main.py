import os
import cv2
import pickle
import shutil

DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.pkl"

# Haar Cascade que já vem com o OpenCV
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def garantir_pastas():
    os.makedirs(DATASET_DIR, exist_ok=True)


def detectar_rosto(gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )
    return faces


def cadastrar_pessoa():
    garantir_pastas()

    nome = input("Nome da pessoa para cadastrar: ").strip()
    if not nome:
        print("Nome inválido.")
        return

    pasta_pessoa = os.path.join(DATASET_DIR, nome)

    if os.path.exists(pasta_pessoa):
        resp = input("Essa pessoa já existe. Deseja apagar e cadastrar de novo? (s/n): ").strip().lower()
        if resp != "s":
            print("Cadastro cancelado.")
            return
        shutil.rmtree(pasta_pessoa)

    os.makedirs(pasta_pessoa, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    print("\nCadastro iniciado.")
    print("Olhe para a câmera. Serão salvas 30 fotos do rosto.")
    print("Pressione Q para cancelar.\n")

    count = 0
    total_fotos = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectar_rosto(gray, face_cascade)

        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # salva uma imagem por frame enquanto houver só um rosto razoável
            if len(faces) == 1 and count < total_fotos:
                arquivo = os.path.join(pasta_pessoa, f"{count+1}.jpg")
                cv2.imwrite(arquivo, rosto)
                count += 1
                cv2.putText(frame, f"Capturadas: {count}/{total_fotos}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Cadastro", frame)
                cv2.waitKey(150)

        cv2.putText(frame, f"Capturadas: {count}/{total_fotos}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Pressione Q para cancelar",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Cadastro", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Cadastro cancelado.")
            break

        if count >= total_fotos:
            print(f"Cadastro de {nome} concluído com {total_fotos} imagens.")
            break

    cap.release()
    cv2.destroyAllWindows()


def treinar_modelo():
    garantir_pastas()

    if not hasattr(cv2, "face"):
        print("Seu OpenCV não tem o módulo cv2.face.")
        print("Instale com: pip install opencv-contrib-python")
        return False

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    imagens = []
    labels = []
    label_map = {}
    current_id = 0

    pessoas = [p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))]

    if not pessoas:
        print("Nenhuma pessoa cadastrada para treinar.")
        return False

    for pessoa in pessoas:
        pasta_pessoa = os.path.join(DATASET_DIR, pessoa)

        if pessoa not in label_map:
            label_map[pessoa] = current_id
            current_id += 1

        label_id = label_map[pessoa]

        for arquivo in os.listdir(pasta_pessoa):
            caminho = os.path.join(pasta_pessoa, arquivo)

            img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = detectar_rosto(img, face_cascade)

            if len(faces) == 0:
                # se a imagem já for só o rosto recortado
                rosto = cv2.resize(img, (200, 200))
                imagens.append(rosto)
                labels.append(label_id)
            else:
                for (x, y, w, h) in faces:
                    rosto = img[y:y+h, x:x+w]
                    rosto = cv2.resize(rosto, (200, 200))
                    imagens.append(rosto)
                    labels.append(label_id)

    if not imagens:
        print("Nenhuma imagem válida encontrada para treino.")
        return False

    recognizer.train(imagens, cv2.UMat if False else __import__("numpy").array(labels))
    recognizer.save(TRAINER_FILE)

    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print("Treinamento concluído com sucesso.")
    return True


def tentar_entrar():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        print("Modelo ainda não treinado.")
        print("Treinando agora...")
        ok = treinar_modelo()
        if not ok:
            return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)

    id_to_name = {v: k for k, v in label_map.items()}

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    print("\nModo entrada.")
    print("Mostre o rosto para a câmera.")
    print("Pressione Q para sair.\n")

    autorizado = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectar_rosto(gray, face_cascade)

        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            label_id, confidence = recognizer.predict(rosto)

            # No LBPH, menor confidence costuma ser melhor
            if confidence < 60:
                nome = id_to_name.get(label_id, "Desconhecido")
                texto = f"Autorizado: {nome} ({confidence:.1f})"
                autorizado = True
            else:
                texto = f"Nao autorizado ({confidence:.1f})"

            cor = (0, 255, 0) if autorizado else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(frame, texto, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        cv2.imshow("Entrada", frame)

        if autorizado:
            print("✅ Acesso autorizado.")
            cv2.waitKey(1500)
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Encerrado.")
            break

    cap.release()
    cv2.destroyAllWindows()


def menu():
    while True:
        print("\n=== SISTEMA DE ACESSO FACIAL ===")
        print("1 - Cadastrar pessoa")
        print("2 - Treinar modelo")
        print("3 - Tentar entrar")
        print("4 - Sair")

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            cadastrar_pessoa()
        elif opcao == "2":
            treinar_modelo()
        elif opcao == "3":
            tentar_entrar()
        elif opcao == "4":
            print("Saindo...")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    menu()