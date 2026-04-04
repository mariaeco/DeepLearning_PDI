import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def carregar_treino(csv_path: str):
    df = pd.read_csv(csv_path)
    X = np.array(df.drop(columns=["target", "Unnamed: 0"], errors="ignore"))
    y = np.array(df["target"])

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y_enc)
    return clf, encoder


def detectar_faces(caminho_imagem: str, tamanho=(160, 160)):
    detector = MTCNN()
    imagem = Image.open(caminho_imagem).convert("RGB")
    array_img = np.asarray(imagem)
    resultados = detector.detect_faces(array_img)

    faces = []
    caixas = []
    for r in resultados:
        x, y, w, h = r["box"]
        x, y = abs(x), abs(y)
        x2, y2 = x + w, y + h

        x = max(0, x)
        y = max(0, y)
        x2 = min(array_img.shape[1], x2)
        y2 = min(array_img.shape[0], y2)
        if x2 <= x or y2 <= y:
            continue

        face = array_img[y:y2, x:x2]
        face_img = Image.fromarray(face).resize(tamanho)
        faces.append(np.asarray(face_img))
        caixas.append((x, y, x2, y2))

    return imagem, faces, caixas


def identificar_pessoas(
    caminho_imagem: str,
    csv_treino: str = "faces.csv",
    limiar: float = 0.65,
    saida: str | None = None,
):
    clf, encoder = carregar_treino(csv_treino)
    imagem, faces, caixas = detectar_faces(caminho_imagem)

    if not faces:
        print("Nenhuma face detectada na imagem.")
        return

    embedder = FaceNet()
    embeddings = embedder.embeddings(np.asarray(faces))
    preds = clf.predict(embeddings)
    probs = clf.predict_proba(embeddings)

    draw = ImageDraw.Draw(imagem)
    print(f"Faces detectadas: {len(faces)}")

    for i, (pred, prob, box) in enumerate(zip(preds, probs, caixas), start=1):
        conf = float(np.max(prob))
        nome = encoder.inverse_transform([pred])[0]
        if conf < limiar:
            nome = "Desconhecido"

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, max(0, y1 - 15)), f"{nome} ({conf:.2f})", fill="lime")
        print(f"Face {i}: {nome} | confianca={conf:.3f}")

    if saida is None:
        p = Path(caminho_imagem)
        saida = str(p.with_name(f"{p.stem}_resultado{p.suffix}"))

    imagem.save(saida)
    print(f"Imagem com resultado salva em: {saida}")
    imagem.show()


def escolher_imagem():
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        caminho = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        root.destroy()
        return caminho
    except Exception:
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identifica pessoas em uma foto.")
    parser.add_argument(
        "imagem",
        nargs="?",
        default=None,
        help="Caminho da imagem. Se nao informar, abre seletor de arquivo.",
    )

    args = parser.parse_args()
    imagem_path = args.imagem if args.imagem else escolher_imagem()

    if not imagem_path:
        print("Nenhuma imagem selecionada.")
    else:
        identificar_pessoas(imagem_path)
