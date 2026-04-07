"""
Alternativa ao identificador_dense.py: treina com imagens (subpastas = pessoas),
como treinar_cnn_faces_zero.py / fluxo CNN sobre pixels — sem embeddings FaceNet.

Fluxo: carrega faces/, treina CNN (160x160), classifica foto nova com MTCNN + modelo.
Grava modelo_dense_imagens.keras e dense_imagens_classes.json. Para só classificar
sem treinar de novo, use classificar_dense_imagens.py.
Não altera main.py.
"""

from __future__ import annotations

import json
import warnings
from os import environ
from pathlib import Path

environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
environ.setdefault("GRPC_VERBOSITY", "ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from mtcnn import MTCNN
from tensorflow import keras
from tensorflow.keras import layers

IMG = 160
OUT_MODEL = "modelo_dense_imagens.keras"
OUT_META = "dense_imagens_classes.json"
OUT_PLOT = "dense_imagens_curvas.png"


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


def treinar_modelo_imagens(data_dir, epochs = 25, batch_size = 8, val_split = 0.2, seed = 42):
    """Mesma ideia de treinar_cnn_faces_zero.py: CNN do zero em RGB 160x160."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta não existe: {root}")

    train_ds = keras.utils.image_dataset_from_directory(
        str(root),
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(IMG, IMG),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        str(root),
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(IMG, IMG),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    class_names = list(train_ds.class_names)
    n = len(class_names)
    if n < 2:
        raise ValueError("Precisa de pelo menos 2 subpastas (pessoas) na pasta de dados.")

    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
        ]
    )
    train_ds = train_ds.map(
        lambda x, y: (aug(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = keras.Sequential(
        [
            layers.Input(shape=(IMG, IMG, 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(n, activation="softmax"),
        ],
        name="cnn_faces_imagens_dense_alt",
    )
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    plot_treino_validacao(hist, OUT_PLOT)
    print("Gráfico treino/validação salvo em:", OUT_PLOT)

    model.save(OUT_MODEL)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "img_size": IMG}, f, indent=2)
    print(f"Modelo salvo em: {OUT_MODEL}")
    print(f"Classes ({n}): {class_names}")

    return model, class_names


def carregar_modelo_salvo(
    model_path: str | Path = OUT_MODEL,
    meta_path: str | Path = OUT_META,
) -> tuple[keras.Model, list[str], int]:
    """
    Carrega o modelo treinado e os metadados (nomes das classes, tamanho da imagem).
    Use para inferência sem rodar treinar_modelo_imagens.
    """
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path.resolve()}. Treine antes com este script."
        )
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Metadados não encontrados: {meta_path.resolve()}. "
            "Eles são gravados junto com o treino (dense_imagens_classes.json)."
        )
    model = keras.models.load_model(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    class_names = list(meta["class_names"])
    img_size = int(meta.get("img_size", IMG))
    return model, class_names, img_size


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


def detectar_faces(
    caminho_imagem: str,
    tamanho: tuple[int, int] = (IMG, IMG),
):
    """Recortes alinhados ao tamanho usado no treino (160x160 por padrão)."""
    detector = MTCNN()
    imagem = Image.open(caminho_imagem).convert("RGB")
    array_img = np.asarray(imagem)
    resultados = detector.detect_faces(array_img)

    faces = []
    caixas = []
    w, h = tamanho
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

        # x, y, bw, bh = r["box"]
        # x, y = abs(x), abs(y)
        # x2, y2 = x + bw, y + bh
        # x = max(0, x)
        # y = max(0, y)
        # x2 = min(array_img.shape[1], x2)
        # y2 = min(array_img.shape[0], y2)
        # if x2 <= x or y2 <= y:
        #     continue
        face = array_img[y1:y2, x1:x2]
        face_img = Image.fromarray(face).resize((w, h))
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


def classificar_imagem_aberta(
    caminho_imagem: str,
    model,
    class_names: list[str],
    limiar: float = 0.5,
    salvar: str | None = None,
    mostrar_plot: bool = True,
    img_size: int | None = None,
):
    """img_size: lado do quadrado de face (deve bater com o treino); None usa IMG."""
    t = img_size if img_size is not None else IMG
    imagem, faces, caixas = detectar_faces(caminho_imagem, tamanho=(t, t))
    if not faces:
        print("Nenhuma face detectada na imagem.")
        return

    labels_arr = np.array(class_names)
    batch = np.stack(faces, axis=0).astype("float32")
    probs = model.predict(batch, verbose=0)
    font = _font_rotulo(22)

    for prob_row, box in zip(probs, caixas):
        class_index = int(np.argmax(prob_row))
        class_probability = float(prob_row[class_index] * 100.0)
        nome = class_names[class_index]
        if prob_row[class_index] < limiar:
            nome = "Desconhecido"

        print("Predicted: %s (%.3f%%)" % (nome, class_probability))
        print("Top 10:")
        top_10_prediction(prob_row, labels_arr)

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
        plt.figure(figsize=(14, 10))
        plt.imshow(np.asarray(imagem))
        plt.title("Faces detectadas")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if salvar is None:
        p = Path(caminho_imagem)
        salvar = str(p.with_name(f"{p.stem}_resultado_dense_imagens{p.suffix}"))
    imagem.save(salvar)
    print("Imagem anotada salva em:", salvar)


if __name__ == "__main__":
    data_dir = "faces"
    epochs = 25
    batch_size = 32
    val_split = 0.2
    limiar = 0.9
    mostrar_plot = True
    imagem = None  # ex.: "minha_foto.jpg" ou None para abrir o seletor

    print("Treinando CNN nas imagens em", data_dir, "...")
    model, class_names = treinar_modelo_imagens(
        data_dir,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
    )

    caminho = imagem if imagem else escolher_imagem()
    if not caminho:
        print("Nenhuma imagem selecionada; treino concluído.")
    else:
        print("Classificando:", caminho)
        classificar_imagem_aberta(
            caminho,
            model,
            class_names,
            limiar=limiar,
            mostrar_plot=mostrar_plot,
        )
