from pathlib import Path

from document_service import processar_documento


if __name__ == "__main__":
    pasta_imgs = Path("imgs")
    extensoes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    if not pasta_imgs.exists() or not pasta_imgs.is_dir():
        print("Pasta imgs nao encontrada.")
    else:
        imagens = sorted(
            [
                caminho
                for caminho in pasta_imgs.iterdir()
                if caminho.is_file() and caminho.suffix.lower() in extensoes
            ]
        )

        if not imagens:
            print("Nenhuma imagem encontrada na pasta imgs.")
        else:
            for caminho_imagem in imagens:
                dados = processar_documento(str(caminho_imagem))
                print("\n---------------")
                print(f"Arquivo: {caminho_imagem.name}\n\n")
                print(dados)