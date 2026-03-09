import matplotlib.pyplot as plt
from PIL import Image
import os

from atrous_correlation import Atrous
import json

# Detecção automatica do diretório
base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, "input")

image1_name = "Shapes.png"
image2_name = "testpat.1k.color2.tif"

image1_path = os.path.join(folder_path, image1_name)
image2_path = os.path.join(folder_path, image2_name)

# Using PIL (Python Imaging Library)
def read_with_pil():
    try:
        img1_pil = Image.open(image1_path)
        print(f"Lido {image1_name}")
        print(f"  Formato: {img1_pil.format}")
        print(f"  Tamanho: {img1_pil.size}")
        print(f"  Modo: {img1_pil.mode}")
        
        img2_pil = Image.open(image2_path)
        print(f"Lido {image2_name}")
        print(f"  Formato: {img2_pil.format}")
        print(f"  Tamanho: {img2_pil.size}")
        print(f"  Modo: {img2_pil.mode}")
        
        return img1_pil, img2_pil
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return None, None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None, None

def display_images(img1, img2, title1="Image 1", title2="Image 2"):
    if img1 is None and img2 is None:
        print("Sem imagens a mostrar")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    if img1 is not None:
        ax1.imshow(img1)
        ax1.set_title(f"{title1}\n{image1_name}")
        ax1.axis('off')
    else:
        ax1.set_title(f"{title1}\n(No image)")
        ax1.axis('off')

    if img2 is not None:
        ax2.imshow(img2)
        ax2.set_title(f"{title2}\n{image2_name}")
        ax2.axis('off')
    else:
        ax2.set_title(f"{title2}\n(No image)")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=" * 50)
    print("PROGRAMA DE LEITURA DE IMAGENS")
    print("=" * 50)
    
    print(f"\Checando pasta: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Pasta nao encontrada")
        print(f"Diretorio atual: {os.getcwd()}")
        
        # Try to find the correct path
        possible_base = r"C:\Users\Mário\Documents"
        if os.path.exists(possible_base):
            print(f"\nDoc base existe: {possible_base}")
            # List contents of Documents folder
            try:
                docs_contents = os.listdir(possible_base)
                print("Conteudo documentos::")
                for item in docs_contents:
                    print(f"  - {item}")
                    if "EngenhariaComputacaoUFPB" in item:
                        print(f"    ACHOU: {item}")
            except Exception as e:
                print(f"Erro listando documentos: {e}")
    else:
        print(f"Pasta encontrada: {folder_path}")
        
        # List files in the folder
        print("\nArquivos na pasta:")
        try:
            files = os.listdir(folder_path)
            if files:
                for file in files:
                    print(f"  - {file}")
                    # Check if our target files are there
                    if file == image1_name:
                        print(f"    Achou: {image1_name}")
                    if file == image2_name:
                        print(f"    Achou: {image2_name}")
            else:
                print("  (pasta vazia)")
        except Exception as e:
            print(f"Erro de lista: {e}")
    
    print("\n" + "=" * 50)
    
    img1_pil, img2_pil = read_with_pil()

    if img1_pil is not None or img2_pil is not None:
        display_images(img1_pil, img2_pil, "PDI - Shapes", "PDI - Test Pattern")

    print("\n" + "=" * 50)
    print("INFO ADICIONAL")
    print("=" * 50)
    
    if img1_pil is not None and img2_pil is not None:
        print("\nDetalhes:")
        print(f"Shapes.png - Modo: {img1_pil.mode}, Tamanho: {img1_pil.size}")
        print(f"testpat.1k.color2.tif - Modo: {img2_pil.mode}, Tamanho: {img2_pil.size}")

    # ------------ MASCARA --------------------------------------------------
    maskname = 'sobelHorizon'

    # ---- fazendo para imagen 1 --------------------------------------------
    image = img1_pil
    size = image.size
    pixel = list(image.getdata()) 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "params", f"{maskname}.json")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = data[maskname]
    atrous = Atrous(name=maskname, pixels=pixel,  params=params, size_r=size[1], size_c=size[0])
    atrous.print_Canal()
 
    # Salvar imagem RGB completa
    file_name = f'img1Final_{maskname}_D{params["dil_rate"]}_S{params["stride"]}_{params["activation"]}.png'
    atrous.salvar_imagem(nome_arquivo=file_name, pasta_saida="output")

    # Salvar canais separados
    atrous.salvar_canais_separados(
        nome_base=f'img1Canais_{maskname}_D{params["dil_rate"]}_S{params["stride"]}_{params["activation"]}',
        pasta_saida="output")

    img_saida = atrous.print_Imagem_final(image)


    # ---- fazendo para imagen 2 --------------------------------------------
    image = img2_pil
    size = image.size
    pixel = list(image.getdata()) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "params", f"{maskname}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    params = data[maskname]
    atrous = Atrous(name=maskname, pixels=pixel, params=params, size_r=size[1], size_c=size[0])

    atrous.print_Canal()

    # Salvar imagem RGB completa
    file_name = f'img2Final_{maskname}_D{params["dil_rate"]}_S{params["stride"]}_{params["activation"]}.png'
    atrous.salvar_imagem(nome_arquivo=file_name, pasta_saida="output")

    # Salvar canais separados
    atrous.salvar_canais_separados(
        nome_base=f'img2Canais_{maskname}_D{params["dil_rate"]}_S{params["stride"]}_{params["activation"]}',
        pasta_saida="output"
    )

    img_saida = atrous.print_Imagem_final(image)
    
