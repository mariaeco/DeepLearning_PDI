"""
Inferência apenas: carrega modelo_dense_imagens.keras + dense_imagens_classes.json
sem treinar. Rode antes identificador_dense_imagens.py para gerar esses arquivos.

Uso: ajuste as variáveis abaixo ou execute e escolha a imagem no seletor.
  python classificar_dense_imagens.py
"""

from __future__ import annotations

from pathlib import Path

from modelo_dense_imagens import (
    OUT_META,
    OUT_MODEL,
    carregar_modelo_salvo,
    classificar_imagem_aberta,
    escolher_imagem,
)


if __name__ == "__main__":
    modelo_path = OUT_MODEL
    meta_path = OUT_META
    limiar = 0.5
    mostrar_plot = True
    imagem = None  # ex.: "minha_foto.jpg" ou None para abrir o seletor

    model, class_names, img_size = carregar_modelo_salvo(modelo_path, meta_path)

    caminho = imagem if imagem else escolher_imagem()
    if not caminho:
        print("Nenhuma imagem selecionada.")
    elif not Path(caminho).is_file():
        print(f"Arquivo não encontrado: {caminho}")
    else:
        print("Classificando:", caminho)
        classificar_imagem_aberta(
            caminho,
            model,
            class_names,
            limiar=limiar,
            mostrar_plot=mostrar_plot,
            img_size=img_size,
        )
