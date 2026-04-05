'''
CADASTRA O ALUNO COM NOME E FOTO
A FOTO PODE SER TIRADA DA WEBCAM OU VIA UPLOAD
AO FINAL É SALVO O EMBEDDING DA FOTO DO ALUNO EM faces.csv
SE FOR FORNECIDO MAIS DE UMA FOTO DO ALUNO, PODE TER MAIS DE UM EMBEDDING
A IDENTIFICAÇÃO FUNCIONA COM APENAS UMA FOTO, MAS SE TIVER MAIS DE UMA ELE COMPARARÁ COM AMBAS.

'''


from __future__ import annotations

import warnings
from os import environ, listdir, makedirs, rename

# Antes de carregar TensorFlow (MTCNN) e face_recognition: reduz ruído no terminal.
environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
environ.setdefault("GRPC_VERBOSITY", "ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import csv
from os.path import basename, isdir, isfile, join, splitext
from pathlib import Path
import cv2
import face_recognition
import tkinter as tk
from tkinter import filedialog
from mtcnn import MTCNN #para extrair faces de fotos
from PIL import Image, ImageDraw #para extrair faces de fotos
from numpy import asarray

import shutil
from datetime import datetime
from pathlib import Path

def get_foto(dir_source):
    """
    Capturar ou fazer o upload de uma foto.
    """
    nome_aluno = input("Digite o nome do aluno: ").strip()
    if not nome_aluno:
        print("Nenhum nome informado.")
        return None

    filename = f"{nome_aluno}.jpg"
    makedirs(dir_source, exist_ok=True)
    path = join(dir_source, filename)

    print("\nComo deseja fornecer a foto do aluno?")
    print("  1 - Capturar pela webcam")
    print("  2 - Escolher imagem no computador")
    escolha = input("Digite 1 ou 2: ").strip()

    if escolha == "1":
        return _capturar_webcam(path), nome_aluno
    if escolha == "2":
        return _escolher_arquivo_imagem(path), nome_aluno

    print("Opção inválida.")
    return None


def _escolher_arquivo_imagem(path) -> str | None:

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        caminho = filedialog.askopenfilename(
            title="Selecione a foto do aluno",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )

        if not caminho:
            print("Nenhum arquivo selecionado.")
            return None

        shutil.copy2(caminho, path)
        print(f"Foto guardada em: {path}")
        return str(path)

    finally:
        root.destroy()


def _capturar_webcam(path) -> str | None:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível abrir a webcam (índice 0).")
        return None

    print(
        "\nWebcam: centralize o rosto.\n"
        "  [ESPAÇO] capturar (deve haver exatamente um rosto)\n"
        "  [Q] cancelar\n"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame da câmera.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
            locs = face_recognition.face_locations(small, model="hog")

            display = frame.copy()
            scale = 4
            for top, right, bottom, left in locs:
                cv2.rectangle(
                    display,
                    (left * scale, top * scale),
                    (right * scale, bottom * scale),
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Registrar aluno - webcam", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                # Mesma detecção já feita em `small` neste frame (evita rodar HOG de novo no rgb).
                if not locs:
                    print("Nenhum rosto detectado. Ajuste e tente de novo.")
                    continue
                if len(locs) > 1:
                    print("Mais de um rosto na imagem. Deixe só o aluno visível.")
                    continue

                if not cv2.imwrite(path, frame):
                    print("Erro ao salvar a imagem.")
                    return None
                print(f"Foto capturada: {path}")
                return path
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None


def _nome_ja_em_frequencia(freq_path: str, nome: str) -> bool:
    if not isfile(freq_path):
        return False
    chave = nome.strip().lower()
    with open(freq_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            if i == 0 and row[0].strip().lower() == "nome":
                continue
            if row[0].strip().lower() == chave:
                return True
    return False


def encodding_aluno(caminho_foto_aluno, nome_aluno):
    """
    Carrega a foto de um aluno, valida se há um rosto e retorna o encoding.
    """
    if not caminho_foto_aluno:
        print("ERRO: caminho da foto vazio.")
        return None
    imagem_aluno = face_recognition.load_image_file(caminho_foto_aluno)
    encodings_face = face_recognition.face_encodings(imagem_aluno)

    if not encodings_face:
        print(f"ERRO: Nenhum rosto foi encontrado na foto do aluno: {caminho_foto_aluno}")
        return None
    else:
        enc = encodings_face[0]
        csv_path = join(Path(__file__).resolve().parent, "faces.csv")
        freq_path = join(Path(__file__).resolve().parent, "frequencia.csv")
        row = [nome_aluno] + enc.tolist()
        header = ["target"] + [f"e{i}" for i in range(len(enc))]

        novo = not isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if novo:
                writer.writerow(header)
            writer.writerow(row)

        if _nome_ja_em_frequencia(freq_path, nome_aluno):
            print(f"Aviso: '{nome_aluno}' já cadastrado em frequencia.csv.")
        else:
            freq_novo = not isfile(freq_path)
            with open(freq_path, "a", newline="", encoding="utf-8") as file:
                freq_writer = csv.writer(file)
                if freq_novo:
                    freq_writer.writerow(["nome"])
                freq_writer.writerow([nome_aluno])

        print(f"✅ Aluno registrado com sucesso a partir de: {caminho_foto_aluno}")

        return encodings_face[0]



if __name__ == "__main__":
    diretorio_base = "alunos_cadastrados"
    resultado = get_foto(diretorio_base)
    if resultado is None:
        print("Cadastro cancelado ou inválido.")
    else:
        foto_aluno, nome_aluno = resultado
        aluno = encodding_aluno(foto_aluno, nome_aluno)
