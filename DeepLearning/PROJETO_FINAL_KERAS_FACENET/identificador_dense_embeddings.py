"""
Um único fluxo (estilo notebook): carrega embeddings do CSV, divide treino/teste,
treina classificador denso 64 -> 32 -> softmax, avalia, e classifica uma foto que você abrir.
Não altera main.py.
"""

from __future__ import annotations

import warnings
from os import environ, listdir, makedirs, rename

# Antes de carregar TensorFlow (MTCNN) e face_recognition: reduz ruído no terminal.
environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
environ.setdefault("GRPC_VERBOSITY", "ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")



from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

OUT_PLOT = "dense_embeddings_curvas.png"


def plot_treino_validacao(history, path: str) -> None:
    """Loss e acurácia por época (treino vs validação) para inspecionar overfitting."""
    h = history.history
    ep = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(ep, h["loss"], label="treino")
    ax1.plot(ep, h["val_loss"], label="validação")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss — treino vs validação")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, h["accuracy"], label="treino")
    ax2.plot(ep, h["val_accuracy"], label="validação")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.set_title("Acurácia — treino vs validação")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.02)

    fig.suptitle(
        "Overfitting: se treino sobe muito e validação estagna ou loss de validação sobe, há overfitting"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def _font_rotulo(tamanho: int = 22):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def carregar_embeddings_csv(csv_path, test_size = 0.2, random_state = 0):
    '''Os embedding sao os dados para o treino'''
    df = pd.read_csv(csv_path)
    X = np.array(df.drop(columns=["target", "Unnamed: 0"], errors="ignore"))
    y = np.array(df["target"])
    try:
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    return train_X, test_X, train_y, test_y


def treinar_classificador_vetores(train_X,train_y, test_X, test_y, epochs = 15, batch_size = 10):
    encoder = LabelEncoder()
    train_y_enc = encoder.fit_transform(train_y)
    test_y_enc = encoder.transform(test_y)
    num_classes = len(encoder.classes_)

    model_vector_classifier = models.Sequential(
        [
            layers.Input(shape=(train_X.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model_vector_classifier.summary()
    model_vector_classifier.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model_vector_classifier.fit(
        train_X,
        train_y_enc,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_X, test_y_enc),
    )
    plot_treino_validacao(hist, OUT_PLOT)
    print("Gráfico treino/validação salvo em:", OUT_PLOT)

    _, acc_train = model_vector_classifier.evaluate(train_X, train_y_enc, verbose=0)
    _, acc_test = model_vector_classifier.evaluate(test_X, test_y_enc, verbose=0)
    print(
        "Accuracy: train=%.3f, test=%.3f"
        % (acc_train * 100.0, acc_test * 100.0)
    )

    return model_vector_classifier, encoder


def top_10_prediction(prob, labels) -> None:
    sorted_rank = (-prob).argsort()[:10]
    label_prob = {}
    for index in sorted_rank:
        class_ = labels[index]
        probability = prob[index]
        label_prob[str(class_).replace(" ", "_")] = probability * 100.0

    s = sorted(label_prob.items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in s:
        nome = k.replace("pins_", "").replace("_", " ")
        print(nome, f"{v:.2f}%")


def detectar_faces(caminho_imagem, tamanho=(250, 250)):
    detector = MTCNN()
    imagem = Image.open(caminho_imagem).convert("RGB")
    array_img = np.asarray(imagem)
    resultados = detector.detect_faces(array_img)

    faces = []
    caixas = []
    for r in resultados:
        x1, y1, width, height = r["box"]
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # Expande a caixa: mais em cima (cabelo), pouco nas laterais e embaixo
        h_img, w_img = array_img.shape[0], array_img.shape[1]
        dy_topo = int(height * 0.35)
        dx_lado = int(width * 0.10)
        dy_baixo = int(height * 0.05)
        x1 = max(0, x1 - dx_lado)
        x2 = min(w_img, x2 + dx_lado)
        y1 = max(0, y1 - dy_topo)
        y2 = min(h_img, y2 + dy_baixo)


        # x, y, w, h = r["box"]
        # x, y = abs(x), abs(y)
        # x2, y2 = x + w, y + h

        # x = max(0, x)
        # y = max(0, y)
        # x2 = min(array_img.shape[1], x2)
        # y2 = min(array_img.shape[0], y2)
        # if x2 <= x or y2 <= y:
        #     continue

        face = array_img[y1:y2, x1:x2]
        face_img = Image.fromarray(face).resize(tamanho)
        faces.append(np.asarray(face_img))
        caixas.append((x1, y1, x2, y2))

    return imagem, faces, caixas


def escolher_imagem() -> str:
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        caminho = filedialog.askopenfilename(
            title="Selecione uma imagem para classificar",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        root.destroy()
        return caminho
    except Exception:
        return ""


def classificar_imagem_aberta(caminho_imagem,  model_vector_classifier, encoder = LabelEncoder, limiar = 0.99,  salvar = None,   mostrar_plot = True):
    imagem, faces, caixas = detectar_faces(caminho_imagem)
    if not faces:
        print("Nenhuma face detectada na imagem.")
        return

    embedder = FaceNet()
    emb = embedder.embeddings(np.asarray(faces))
    font = _font_rotulo(22)

    for e, box in zip(emb, caixas):
        samples = np.expand_dims(e, axis=0)
        yhat_prob = model_vector_classifier.predict(samples, verbose=0)
        prob_row = yhat_prob[0]
        class_index = int(np.argmax(prob_row))
        class_probability = float(prob_row[class_index] * 100.0)
        predict_names = encoder.inverse_transform([class_index])
        nome = predict_names[0]
        if prob_row[class_index] < limiar:
            nome = "Desconhecido"

        print("Predicted: %s (%.3f%%)" % (nome, class_probability))
        print("Top 10:")
        top_10_prediction(prob_row, encoder.classes_)

        draw = ImageDraw.Draw(imagem)
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        texto = "%s (%.1f%%)" % (nome, class_probability)
        _, top, _, bottom = draw.textbbox((0, 0), texto, font=font)
        altura_txt = bottom - top
        draw.text(
            (x1, max(0, y1 - altura_txt - 4)),
            texto,
            fill="lime",
            font=font,
        )

    if mostrar_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(np.asarray(imagem))
        plt.title("Faces detectadas")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if salvar is None:
        p = Path(caminho_imagem)
        salvar = str(p.with_name(f"{p.stem}_resultado_dense{p.suffix}"))
    imagem.save(salvar)
    print("Imagem anotada salva em:", salvar)


if __name__ == "__main__":
    csv_path = "faces.csv"
    epochs = 15
    batch_size = 10
    test_size = 0.2
    limiar = 0.99
    mostrar_plot = True
    imagem = None  # ex.: "minha_foto.jpg" ou None para abrir o seletor

    print("Carregando embeddings e dividindo treino/teste...")
    train_X, test_X, train_y, test_y = carregar_embeddings_csv(
        csv_path, test_size=test_size, random_state=0
    )

    print("Treinando model_vector_classifier (do zero)...")
    model_vector_classifier, encoder = treinar_classificador_vetores(
        train_X,
        train_y,
        test_X,
        test_y,
        epochs=epochs,
        batch_size=batch_size,
    )

    caminho = imagem if imagem else escolher_imagem()
    if not caminho:
        print("Nenhuma imagem selecionada; treino concluido.")
    else:
        print("Classificando:", caminho)
        classificar_imagem_aberta(
            caminho,
            model_vector_classifier,
            encoder,
            limiar=limiar,
            mostrar_plot=mostrar_plot,
        )
