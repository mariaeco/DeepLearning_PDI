"""
Reduz resolução das fotos (maior lado limitado) antes de rodar extract_faces.py.
Evita imagens gigantes que fazem o MTCNN estourar memória no NMS.

Uso típico (na pasta do projeto):
  python resize_images_before_extract.py
  python resize_images_before_extract.py --max-side 1024 imgs imagens_teste
  python resize_images_before_extract.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _save_image(img: Image.Image, path: Path, jpeg_quality: int) -> None:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        rgb.save(path, "JPEG", quality=jpeg_quality, optimize=True)
    elif suf == ".png":
        img.save(path, "PNG", optimize=True)
    elif suf == ".webp":
        img.save(path, "WEBP", quality=jpeg_quality)
    else:
        img.save(path)


def resize_one(path: Path, max_side: int, jpeg_quality: int, dry_run: bool) -> bool:
    """
    Se max(largura, altura) > max_side, redimensiona mantendo proporção.
    Retorna True se alterou (ou simulou alteração).
    """
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        m = max(w, h)
        if m <= max_side:
            return False

        scale = max_side / m
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if dry_run:
            print(f"[dry-run] {path}  {w}x{h} -> {new_w}x{new_h}")
            return True

        resized = im.resize((new_w, new_h), _LANCZOS)
        _save_image(resized, path, jpeg_quality)
        print(f"OK {path}  {w}x{h} -> {new_w}x{new_h}")
        return True


def process_roots(roots: list[Path], max_side: int, jpeg_quality: int, dry_run: bool) -> tuple[int, int]:
    changed = 0
    skipped_small = 0
    for root in roots:
        if not root.is_dir():
            print(f"Aviso: pasta não encontrada (ignorada): {root}")
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUFFIXES:
                continue
            try:
                if resize_one(path, max_side, jpeg_quality, dry_run):
                    changed += 1
                else:
                    skipped_small += 1
            except OSError as e:
                print(f"Erro ao ler/gravar {path}: {e}")
    return changed, skipped_small


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduz imagens pelo maior lado antes da extração de faces (MTCNN)."
    )
    parser.add_argument(
        "pastas",
        nargs="*",
        default=["imgs", "imagens_teste"],
        help="Pastas com subpastas por pessoa (padrão: imgs imagens_teste).",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1200,
        help="Tamanho máximo do maior lado em pixels (largura ou altura).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="Qualidade JPEG/WebP ao salvar (1-100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só mostra o que seria redimensionado, sem gravar.",
    )
    args = parser.parse_args()
    roots = [Path(p) for p in args.pastas]

    print(f"max_side={args.max_side}  pastas={roots}  dry_run={args.dry_run}")
    changed, skipped = process_roots(
        roots, args.max_side, args.jpeg_quality, args.dry_run
    )
    print(
        f"Concluído: {changed} imagens redimensionadas"
        + (f", {skipped} já menores ou iguais ao limite." if skipped else ".")
    )


if __name__ == "__main__":
    main()
