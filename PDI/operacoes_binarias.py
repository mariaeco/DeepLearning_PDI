import cv2
import numpy as np
from pathlib import Path


def channel_blend_lighten(a, b):
    return np.maximum(a, b).astype(np.uint8)


def channel_blend_darken(a, b):
    return np.minimum(a, b).astype(np.uint8)


def channel_blend_multiply(a, b):
    return ((a.astype(np.uint16) * b.astype(np.uint16)) // 255).astype(np.uint8)


def channel_blend_average(a, b):
    return ((a.astype(np.uint16) + b.astype(np.uint16)) // 2).astype(np.uint8)


def channel_blend_add(a, b):
    soma = a.astype(np.uint16) + b.astype(np.uint16)
    return np.clip(soma, 0, 255).astype(np.uint8)


def channel_blend_subtract(a, b):
    soma = a.astype(np.uint16) + b.astype(np.uint16)
    return np.where(soma < 255, 0, soma - 255).astype(np.uint8)


def rotular(img, texto):
    saida = img.copy()
    cv2.putText(saida, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(saida, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return saida


def main():
    base_dir = Path(__file__).resolve().parent
    img1_path = str(base_dir / "imagem1.jpg")
    img2_path = str(base_dir / "imagem2.jpg")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Erro: nao foi possivel carregar imagem1.jpg e imagem2.jpg.")
        return

    h, w, _ = img1.shape
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    operacoes = {
        "Lighten": channel_blend_lighten,
        "Darken": channel_blend_darken,
        "Multiply": channel_blend_multiply,
        "Average": channel_blend_average,
        "Add": channel_blend_add,
        "Subtract": channel_blend_subtract,
    }

    resultados = []
    for nome, func in operacoes.items():
        mistura = func(img1, img2)
        resultados.append(rotular(mistura, nome))

    linha_1 = np.hstack(resultados[:3])
    linha_2 = np.hstack(resultados[3:])
    painel = np.vstack([linha_1, linha_2])

    escala = 0.35
    painel = cv2.resize(painel, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    cv2.imshow("Operacoes Binarias - 2x3", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
