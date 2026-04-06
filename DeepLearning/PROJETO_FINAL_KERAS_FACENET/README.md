# Reconhecimento facial com FaceNet e alternativas

Projeto de **processamento de imagens**, **extração de faces**, **embeddings** (vetores FaceNet) e **classificação** com scikit-learn e/ou redes densas/CNN no Keras. O fluxo principal segue a ordem indicada em `main.py`.

## Dataset e pastas para treino

Para **treinar** com o conjunto sugerido, baixe as imagens do dataset **Face Recognition** no Kaggle (pasta **Original Images**) e coloque o conteúdo na pasta **`imgs/`** na raiz deste projeto:

- [Face Recognition Dataset — Kaggle (Original Images)](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset/data?select=Original+Images)

O script `extract_faces.py` espera **uma subpasta por pessoa** (nome da classe = nome da pasta), com as fotos **diretamente dentro** dessa subpasta.

**Estrutura em `imgs/` (treino):**

```text
imgs/
├── Nome_da_Pessoa_A/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
├── Nome_da_Pessoa_B/
│   └── ...
└── ...
```

**Estrutura em `imagens_teste/` (validação):** a mesma lógica — **uma pasta por pessoa**, com imagens de teste para a mesma identidade (ou um subconjunto coerente com o treino). O `extract_faces.py` gera `faces_validation/` a partir daqui; use pessoas que também existam em `imgs/` se quiser métricas comparáveis no `main.py`.

```text
imagens_teste/
├── Nome_da_Pessoa_A/
│   └── teste1.jpg
├── Nome_da_Pessoa_B/
│   └── ...
└── ...
```

Se o download do Kaggle vier com uma pasta extra (por exemplo `Original Images/`), **extraia** o que estiver dentro para que `imgs/` contenha só as pastas das pessoas, não um único arquivo zip solto.

## Pré-requisitos

- Python 3 com TensorFlow/Keras, scikit-learn, pandas, numpy, matplotlib, Pillow  
- [MTCNN](https://github.com/ipazc/mtcnn), [keras-facenet](https://github.com/nyoki-mtl/keras-facenet) (embeddings FaceNet)

Ajuste os caminhos nas pastas (`imgs/`, `faces/`, `imagens_teste/`, `faces_validation/`) conforme o seu dataset.

---

## Pipeline principal (treino e avaliação)

Ordem sugerida:

1. **`extract_faces.py`** — prepara o dataset  
2. **`embeddings.py`** — gera vetores FaceNet  
3. **`main.py`** — compara classificadores sobre os embeddings  

### `extract_faces.py`

- Classe **`ProcessImages`**: detector **MTCNN**, recorte da face com **expansão da caixa** (mais acima para cabelo, laterais e parte inferior), redimensionamento **250×250**.
- Para cada imagem válida: salva o recorte e uma versão **espelhada** (`flip*`) para aumentar dados.
- **`load_dir(dir_source, dir_target)`**: percorre subpastas (uma por pessoa), processa todas as imagens e grava em `dir_target`.
- Por padrão no `if __name__ == "__main__"`: `imgs/` → `faces/` e `imagens_teste/` → `faces_validation/` (conjunto de validação para métricas em `main.py`).

### `embeddings.py`

- Carrega todas as faces de cada subpasta de `faces/` (e opcionalmente `faces_validation/`).
- Usa **`FaceNet`** (`keras_facenet`) para gerar **embeddings** a partir dos arrays RGB.
- Exporta **`faces.csv`** e **`faces_validation.csv`**: colunas = componentes do vetor + coluna **`target`** (nome da classe / pasta).

### `main.py`

- Lê `faces.csv` e `faces_validation.csv`, codifica rótulos com **`LabelEncoder`**.
- Avalia no mesmo espaço de embeddings:
  - **KNN** (`k=5`)
  - **SVM**
  - **Rede densa Keras** (entrada = dimensão do embedding → Dense 64 → Dropout → softmax)
- Imprime acurácia de treino e validação e exibe **matrizes de confusão** para cada método.

---

## Identificação em fotos novas

### `identificador.py` (fluxo clássico: embedding + SVM)

- **Treino**: lê `faces.csv`, treina um **`SVC`** linear com **probabilidades** (`probability=True`).
- **Inferência**: MTCNN detecta faces (recorte **160×160**), FaceNet gera embedding, o SVM classifica; desenha caixas e rótulos na imagem.
- **Limiar de confiança** padrão **0.65**: abaixo disso o rótulo vira **"Desconhecido"**.
- Uso: `python identificador.py [caminho_da_imagem]` ou sem argumentos para abrir o seletor de arquivo (Tkinter).
- Saída: imagem `*_resultado.*` no mesmo diretório da entrada.

---

## Alternativas “dense”: `identificador_dense_embeddings` vs `identificador_dense_imagens`

Ambos são **fluxos alternativos** que **não substituem** `main.py`; servem para treinar um classificador próprio e testar numa foto (com gráficos de treino/validação e opção de salvar imagem anotada).

| Aspecto | `identificador_dense_embeddings.py` | `identificador_dense_imagens.py` |
|--------|-------------------------------------|----------------------------------|
| **Entrada de treino** | CSV **`faces.csv`** (já são embeddings FaceNet) | Pasta **`faces/`** (subpastas = pessoas, imagens RGB) |
| **Modelo** | Rede densa no vetor: **64 → 32 → Dropout → softmax** | **CNN** sobre pixels (**160×160**): convoluções + Dense softmax — **sem FaceNet no treino** |
| **FaceNet** | Sim: gera embeddings na **classificação** da foto nova | **Não** no treino; só **MTCNN + redimensionamento** para alinhar ao tamanho da rede |
| **Recorte da face** | Caixa expandida (similar ao `extract_faces`), **250×250** | Recorte padrão MTCNN redimensionado para **160×160** |
| **Artefatos** | Gráfico `dense_embeddings_curvas.png` | `modelo_dense_imagens.keras`, `dense_imagens_classes.json`, `dense_imagens_curvas.png` |
| **Limiar típico** | Mais alto (ex.: **0.99**) no exemplo | Ex.: **0.9** (ajustável) |

Em resumo:

- **`identificador_dense_embeddings.py`**: continua no **espaço de embeddings FaceNet**; só troca o “cabeçote” de classificação por uma **MLP Keras** treinada com split treino/teste a partir do CSV, com curvas de loss/acurácia.
- **`identificador_dense_imagens.py`**: **treina do zero** uma **CNN em imagens brutas** (como um fluxo estilo classificação por pixels), **sem** depender de `embeddings.py` para o aprendizado — aproxima a ideia de treinar uma rede sobre as faces recortadas, não sobre vetores pré-computados.

---

## Resumo visual do fluxo

```text
[Pastas por pessoa em imgs/]
        → extract_faces.py → faces/ (+ flip)
        → embeddings.py → faces.csv (+ faces_validation.csv)
        → main.py (KNN, SVM, Keras nos embeddings)

[Foto nova]
        → identificador.py (FaceNet + SVM treinado em faces.csv)

Alternativas:
        → identificador_dense_embeddings.py (MLP sobre faces.csv + FaceNet na inferência)
        → identificador_dense_imagens.py (CNN treinada em faces/ + inferência sem embedding FaceNet)
```

---

## Arquivos de dados e saída (referência)

| Arquivo / pasta | Papel |
|-----------------|--------|
| `imgs/`, `imagens_teste/` | Imagens brutas por subpasta (pessoa) |
| `faces/`, `faces_validation/` | Faces recortadas geradas pelo `extract_faces.py` |
| `faces.csv`, `faces_validation.csv` | Embeddings + coluna `target` |

Para detalhes de hiperparâmetros (épocas, batch, limiares), veja as variáveis no `if __name__ == "__main__"` de cada script.
