import cv2
import numpy as np
from pathlib import Path

def misturar_imagens(img1_path, img2_path, modo='horizontal_cv'):
    # Carregar imagens
    f1 = cv2.imread(img1_path)
    f2 = cv2.imread(img2_path)
    
    if f1 is None or f2 is None:
        return "Erro: Verifique os caminhos das imagens."

    # Redimensionar f2 para o tamanho de f1 (RxC)
    R, C, _ = f1.shape
    f2 = cv2.resize(f2, (C, R))
    
    # Criar grades de coordenadas i (0 a R-1) e j (0 a C-1)
    i, j = np.ogrid[:R, :C]
    
    # Dicionário de fórmulas para t(i, j)
    formulas = {
        # Média Horizontal (Varia da esquerda para a direita)
        'horiz_esq_dir': j / (C - 1.0),
        'horiz_dir_esq': (C - 1.0 - j) / (C - 1.0),
        
        # Média Vertical (Varia de cima para baixo)
        'vert_cima_baixo': i / (R - 1.0),
        'vert_baixo_cima': (R - 1.0 - i) / (R - 1.0),
        
        # Diagonais Principais (Top-Left <-> Bottom-Right)
        'diag_principal_normal': (i + j) / (R + C - 2.0),
        'diag_principal_inversa': (R + C - 2.0 - (i + j)) / (R + C - 2.0),
        
        # Diagonais Secundárias (Top-Right <-> Bottom-Left)
        'diag_secundaria_normal': (i + (C - 1 - j)) / (R + C - 2.0),
        'diag_secundaria_inversa': (j + (R - 1 - i)) / (R + C - 2.0)
    }

    if modo not in formulas:
        return "Modo inválido."

    # Obter o mapa de pesos t e expandir para 3 canais (RGB)
    t = formulas[modo]
    t_3d = np.repeat(t[:, :, np.newaxis], 3, axis=2)

    # Cálculo da Mistura: f_result = (1-t)*f1 + t*f2
    resultado = (1 - t_3d) * f1 + t_3d * f2
    return resultado.astype(np.uint8)




# --- Exemplo de teste para todas as variações ---
modos = [
    'horiz_esq_dir',
    'horiz_dir_esq',
    'vert_cima_baixo',
    'vert_baixo_cima',
    'diag_principal_normal',
    'diag_principal_inversa',
    'diag_secundaria_normal',
    'diag_secundaria_inversa'
]

base_dir = Path(__file__).resolve().parent
img1 = str(base_dir / 'imagem1.jpg')
img2 = str(base_dir / 'imagem2.jpg')

painel = []

for m in modos:
    res = misturar_imagens(img1, img2, modo=m)
    if isinstance(res, str):
        print(f"[ERRO] {m}: {res}")
        continue
    cv2.putText(
        res,
        m,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    painel.append(res)

if painel:
    # Organiza em 2 linhas x 4 colunas
    linha_1 = np.hstack(painel[:4])
    linha_2 = np.hstack(painel[4:8])
    visao_geral = np.vstack([linha_1, linha_2])
    escala = 0.35  # Reduz o tamanho da visualizacao final
    visao_geral = cv2.resize(
        visao_geral,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_AREA
    )
    cv2.imshow('Comparacao geral', visao_geral)
    cv2.waitKey(0)
    cv2.destroyAllWindows()