
'''
TIRAR UMA FOTO OU CARREGAR FOTO DA TURMA
E VERIFICA QUAIS DOS ALUNOS CADASTRADOS ESTÃO FREQUENTES NA FOTO
SALVA A FREQUENCIA DO DIA
'''
import csv
import cv2
import face_recognition
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont

# Pastas e ficheiros por defeito (evita muitos parâmetros Optional)
DIR_PROJETO = Path(__file__).resolve().parent
PATH_FACES_CSV = DIR_PROJETO / "faces.csv"
PATH_FREQUENCIA_CSV = DIR_PROJETO / "frequencia.csv"
DIR_FOTO_FREQUENCIA = DIR_PROJETO / "foto_frequencia"


def _lista_encodings(encodings_aluno) -> list:
    """Um vetor (128,) ou lista de vetores do mesmo aluno."""
    if isinstance(encodings_aluno, (list, tuple)):
        return list(encodings_aluno)
    return [encodings_aluno]


def melhor_distancia_aluno_foto(encodings_aluno, encodings_foto) -> float:
    """
    Menor distância entre qualquer embedding do aluno e qualquer rosto na foto.
    encodings_foto: lista devolvida por face_recognition.face_encodings na imagem da turma.
    """
    refs = _lista_encodings(encodings_aluno)
    if not refs or not encodings_foto:
        return float("inf")
    melhor = float("inf")
    for enc_ref in refs:
        dists = face_recognition.face_distance(encodings_foto, enc_ref)
        melhor = min(melhor, float(np.min(dists)))
    return melhor


def verificar_presenca(encodings_aluno, caminho_foto_teste, tolerance=0.5):
    """
    Diz se o aluno aparece na foto de teste.

    Na foto podem existir vários rostos: compara o aluno com **todos**; basta um rosto
    bastante parecido (menor distância <= tolerance).

    encodings_aluno pode ser um único vetor (128,) ou **vários** do mesmo nome
    (várias linhas no faces.csv): usa-se a **menor** distância entre qualquer um deles
    e qualquer rosto na foto.
    """
    imagem_teste = face_recognition.load_image_file(caminho_foto_teste)
    encodings_rostos_teste = face_recognition.face_encodings(imagem_teste)

    if not encodings_rostos_teste:
        print("   ⚠️  Nenhum rosto foi encontrado na imagem de teste")
        return False

    return melhor_distancia_aluno_foto(encodings_aluno, encodings_rostos_teste) <= tolerance


def carregar_faces_csv(csv_path):
    """
    Lê faces.csv: coluna 0 = nome, 128 floats seguintes = embedding.
    Várias linhas com o mesmo nome viram vários embeddings da mesma pessoa.
    Ignora linhas inválidas e linha de cabeçalho (ex.: target, e0, … sem números).
    """
    path = Path(csv_path)
    if not path.is_file():
        return {}
    por_nome = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 129:
                continue
            try:
                vec = np.array([float(x) for x in row[1:129]], dtype=np.float64)
            except ValueError:
                continue
            nome = row[0].strip()
            if not nome:
                continue
            por_nome.setdefault(nome, []).append(vec)
    return por_nome



def _ler_frequencia_csv(freq_path):
    """Cabeçalhos + mapa nome -> células das colunas após a primeira (nome)."""
    if not freq_path.is_file():
        return ["nome"], {}
    with open(freq_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ["nome"], {}
    headers = [str(h).strip() for h in rows[0]]
    if not headers:
        headers = ["nome"]
    if not headers[0]:
        headers[0] = "nome"
    n = len(headers)
    data = {}
    for row in rows[1:]:
        if not row or not str(row[0]).strip():
            continue
        nome = str(row[0]).strip()
        vals = list(row[1:n])
        while len(vals) < n - 1:
            vals.append("")
        data[nome] = vals[: n - 1]
    return headers, data


def _escrever_frequencia_csv(freq_path, headers, data, nomes_ordem):
    n_vals = len(headers) - 1
    with open(freq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for nome in nomes_ordem:
            row_vals = data.get(nome, [])
            padded = (list(row_vals) + [""] * n_vals)[:n_vals]
            w.writerow([nome] + padded)


def acrescentar_frequencia_no_csv(resultado, freq_path=None, rotulo_coluna=None):
    """
    Acrescenta uma coluna ao frequencia.csv: cabeçalho = data-hora (ou rotulo_coluna),
    células = presente | ausente por aluno. Mantém colunas anteriores.
    """
    path = Path(freq_path) if freq_path else PATH_FREQUENCIA_CSV
    col = rotulo_coluna or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers, data = _ler_frequencia_csv(path)
    if col in headers:
        col = f"{col}.{datetime.now().microsecond}"
    old_n = len(headers) - 1
    nomes = sorted(set(data.keys()) | set(resultado.keys()))
    for nome in nomes:
        row = (data.get(nome, []) + [""] * old_n)[:old_n]
        ok, _ = resultado.get(nome, (False, float("inf")))
        row.append("presente" if ok else "ausente")
        data[nome] = row
    headers.append(col)
    _escrever_frequencia_csv(path, headers, data, nomes)
    print(f"Frequência registada em {path} (coluna «{col}»).")
    return col


def _embeddings_nomes_planos(por_nome):
    """Uma entrada por linha do CSV: mesmo nome pode repetir com embeddings diferentes."""
    known_enc = []
    known_names = []
    for nome in sorted(por_nome.keys()):
        for vec in por_nome[nome]:
            known_enc.append(vec)
            known_names.append(nome)
    return known_enc, known_names


def _nome_por_menor_distancia(face_encoding, known_encodings, known_names, tolerance):
    if not known_encodings:
        return "Desconhecido", 1.0
    dists = face_recognition.face_distance(known_encodings, face_encoding)
    i = int(np.argmin(dists))
    dist = float(dists[i])
    if dist <= tolerance:
        return known_names[i], dist
    return "Desconhecido", dist


def _fonte_rotulo(tamanho_px):
    candidatos = []
    if sys.platform == "win32":
        w = os.environ.get("WINDIR", r"C:\Windows")
        candidatos = [
            Path(w) / "Fonts" / "arial.ttf",
            Path(w) / "Fonts" / "calibri.ttf",
            Path(w) / "Fonts" / "segoeui.ttf",
        ]
    else:
        candidatos = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
    for fp in candidatos:
        if fp.is_file():
            try:
                return ImageFont.truetype(str(fp), tamanho_px)
            except OSError:
                continue
    return ImageFont.load_default()


def get_foto(dir_source):
    """
    Capturar ou fazer o upload de uma foto.
    """
    print("\nComo deseja fornecer a foto da Frequência?")
    print("  1 - Capturar pela webcam")
    print("  2 - Escolher imagem no computador")
    escolha = input("Digite 1 ou 2: ").strip()

    if escolha == "1":
        return _capturar_webcam(dir_source)
    if escolha == "2":
        return _escolher_arquivo_imagem()

    print("Opção inválida.")
    return None

def _capturar_webcam(dir_source=None):
    pasta = dir_source or str(DIR_FOTO_FREQUENCIA)
    os.makedirs(pasta, exist_ok=True)
    nome_arquivo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(pasta, nome_arquivo)

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

            cv2.imshow("Frequência - webcam", display)
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

                # Grava o frame original (sem retângulos) para usar com face_recognition depois.
                if not cv2.imwrite(path, frame):
                    print("Erro ao salvar a imagem.")
                    return None
                print(f"Foto capturada: {path}")
                return path
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None

def _escolher_arquivo_imagem():
    pasta = DIR_FOTO_FREQUENCIA
    pasta.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        caminho = filedialog.askopenfilename(
            title="Selecione a foto da frequência (turma)",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )

        if not caminho:
            print("Nenhum arquivo selecionado.")
            return None

        ext = Path(caminho).suffix.lower() or ".jpg"
        destino = pasta / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        shutil.copy2(caminho, destino)
        print(f"Foto guardada em: {destino}")
        return str(destino)
    finally:
        root.destroy()



def reconhecer_rostos(
    caminho_foto_teste,
    faces_csv=None,
    frequencia_csv=None,
    rotulo_coluna_frequencia=None,
    tolerance=0.5,
    model="hog",
    saida=None,
    tamanho_fonte=28,
    mostrar_imagem=True,
):
    """
    Usa faces.csv (várias linhas por aluno permitidas): para cada nome, calcula a
    **menor** distância até qualquer rosto na foto da turma. Presente se <= tolerance.
    Desenha retângulos e rótulos na imagem, grava cópia com sufixo _resultado_fr e
    opcionalmente abre o visualizador.
    faces_csv / frequencia_csv: None = PATH_FACES_CSV / PATH_FREQUENCIA_CSV.
    Devolve {nome: (presente, distância_mínima)}.
    """
    csv_path = Path(faces_csv) if faces_csv else PATH_FACES_CSV
    por_nome = carregar_faces_csv(csv_path)
    if not por_nome:
        print(f"Nenhum embedding em {csv_path}")
        return {}

    imagem = face_recognition.load_image_file(caminho_foto_teste)
    locs = face_recognition.face_locations(imagem, model=model)
    encodings_na_foto = face_recognition.face_encodings(imagem, locs)
    if not encodings_na_foto:
        print("Nenhum rosto detectado na foto de frequência.")
        return {}

    known_enc, known_names = _embeddings_nomes_planos(por_nome)
    pil_image = Image.fromarray(imagem)
    draw = ImageDraw.Draw(pil_image)
    fonte = _fonte_rotulo(tamanho_fonte)

    #MOSTRAR NA IMAGEM A DETECÇÃO
    print(f"Faces detectadas na foto: {len(encodings_na_foto)}")
    for i, (enc, loc) in enumerate(zip(encodings_na_foto, locs), start=1):
        top, right, bottom, left = loc
        nome, dist = _nome_por_menor_distancia(enc, known_enc, known_names, tolerance)

        draw.rectangle([left, top, right, bottom], outline="lime", width=3)
        label = f"{nome} ({dist:.2f})"
        bbox = draw.textbbox((0, 0), label, font=fonte)
        altura = bbox[3] - bbox[1]
        ty = max(0, top - altura - 6)
        draw.text(
            (left, ty),
            label,
            fill="lime",
            font=fonte,
            stroke_width=max(1, tamanho_fonte//8),
            stroke_fill="black",
        )
        print(f"   Face {i}: {nome} | distância={dist:.3f}")

    if saida is None:
        p = Path(caminho_foto_teste)
        saida = str(p.with_name(f"{p.stem}_resultado_fr{p.suffix}"))

    pil_image.save(saida)
    print(f"Imagem com resultado salva em: {saida}")
    if mostrar_imagem:
        pil_image.show()

    resultado = {}
    for nome in sorted(por_nome.keys()):
        dist = melhor_distancia_aluno_foto(por_nome[nome], encodings_na_foto)
        ok = dist <= tolerance
        resultado[nome] = (ok, dist)
        status = "presente" if ok else "ausente"
        print(f"   {nome}: {status} (menor distância {dist:.3f})")

    acrescentar_frequencia_no_csv(
        resultado,
        frequencia_csv or PATH_FREQUENCIA_CSV,
        rotulo_coluna_frequencia,
    )
    return resultado



if __name__ == "__main__":
    resultado = get_foto(str(DIR_FOTO_FREQUENCIA))
    if resultado:
        reconhecer_rostos(resultado, PATH_FACES_CSV, PATH_FREQUENCIA_CSV)


