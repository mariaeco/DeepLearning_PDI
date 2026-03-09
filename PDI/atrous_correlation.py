
import json
import os
from re import I
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import masked_array 
from PIL import Image 

class Atrous:

    def __init__(self, name, pixels, params=None, size_r=None, size_c=None):

        # Parâmetros vindos do gauss.json
        self.name = name
        self.pixels = pixels
        self.size_r = size_r
        self.size_c = size_c
        self.mask = params["mask"]

        self.divide = params["divide"]
        self.stride = params["stride"]
        self.dil_rate = params["dil_rate"]
        self.activation_name = params.get("activation", "relu")
        self.pre_expansao_por_canal = {}

        self.R, self.G, self.B = self.get_canal()

        self.RSaida = self.correlacao_Atrous(self.name,self.mask,'R',self.R, self.size_r, self.size_c, rate=self.dil_rate, stride=self.stride)
        self.GSaida = self.correlacao_Atrous(self.name,self.mask,'G',self.G, self.size_r, self.size_c, rate=self.dil_rate, stride=self.stride)
        self.BSaida = self.correlacao_Atrous(self.name,self.mask,'B',self.B, self.size_r, self.size_c, rate=self.dil_rate, stride=self.stride)
        
        if self.name == 'sobelHorizon' or self.name == 'sobelVert':
            self.plotar_histograma_rgb_final(self.pre_expansao_por_canal.get('R'),
                  self.pre_expansao_por_canal.get('G'),
                  self.pre_expansao_por_canal.get('B'),'Antes da Expansão')
            self.plotar_histogramas_final()
            self.plotar_histograma_rgb_final(self.RSaida,self.RSaida,self.RSaida,'Após Expansão')


    def get_canal(self):
        R_chanel = []
        G_chanel = []
        B_chanel = [] 
        for px in self.pixels:
            R_chanel.append(px[0])
            G_chanel.append(px[1])
            B_chanel.append(px[2])

        return R_chanel, G_chanel, B_chanel

                
    def correlacao_Atrous(self, name, mask, canal, pixels_canal, size_r, size_c, rate, stride):

        pixels_arr = np.array(pixels_canal, dtype=float).reshape(size_r, size_c)
        mask_arr = np.array(mask, dtype=float)

        f_r, f_c = mask_arr.shape
        
        self.saida = []
        f_eff_r = ((f_r - 1)*rate) + 1 
        f_eff_c = ((f_c - 1)*rate) + 1 
        dimensao_saida_c = np.int32(np.floor(((size_c - f_eff_c)/stride)+1))
        dimensao_saida_r = np.int32(np.floor(((size_r - f_eff_r)/stride)+1))
        print(f'Tamanho da imagem {size_r}x{size_c}, Máscara {f_r}x{f_c}, Imagem de Saída {dimensao_saida_r}x{dimensao_saida_c}')

        for i_out in range(dimensao_saida_r):
            for j_out in range(dimensao_saida_c):
                soma = 0
                i = i_out * stride
                j = j_out * stride
                for u in range(f_r):
                    for v in range(f_c):
                        soma += (
                            pixels_arr[i + u*rate, j + v*rate] * mask_arr[u, v]
                        )
                valor = soma/self.divide 
                self.saida.append(valor)

        self.saida = np.array(self.saida, dtype=float).reshape(dimensao_saida_r, dimensao_saida_c)
        
        if name=='sobelHorizon' or name == 'sobelVert':
            self.saida = np.abs(self.saida)
            self.pre_expansao_por_canal[canal] = self.saida.copy()
            self.saida = self.expansao_histograma(self.saida, 0, 255)
            
        else:
            self.saida = self.get_activation(self.saida, name=self.activation_name)
        return self.saida


    def plotar_histogramas_final(self):
        """
        Plota histogramas dos três canais RGB apenas para filtros Sobel.
        """
        # Mostrar histogramas apenas para filtros Sobel
        if self.name != 'sobelHorizon' and self.name != 'sobelVert':
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        cores = ['red', 'green', 'blue']
        canais = [self.RSaida, self.GSaida, self.BSaida]
        nomes_canais = ['R', 'G', 'B']
        
        for idx, (ax, canal, cor, nome) in enumerate(zip(axes, canais, cores, nomes_canais)):
            ax.hist(canal.flatten(), bins=50, color=cor, edgecolor='black', alpha=0.7)
            ax.set_title(f'Histograma Final - Canal {nome}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Intensidade de Pixel')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Adicionar informações estatísticas
            stats = f'Min: {canal.min():.2f}\nMax: {canal.max():.2f}\nMédia: {canal.mean():.2f}'
            ax.text(0.98, 0.97, stats, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        plt.show()

    def plotar_histograma_rgb_final(self, canalR, canalG, canalB, name):
        """
        Plota o histograma da imagem final RGB combinada.
        Mostra a distribuição de intensidades apenas para filtros Sobel.
        """
        # Mostrar histograma apenas para filtros Sobel
        if self.name != 'sobelHorizon' and self.name != 'sobelVert':
            return
        
        imagem_rgb = np.stack((canalR, canalG, canalB), axis=2)
        imagem_rgb = np.clip(imagem_rgb, 0, 255).astype(np.uint8)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Plotar histograma combinado de todos os pixels
        ax.hist(imagem_rgb.flatten(), bins=256, color='gray', edgecolor='black', alpha=0.7, range=(0, 256),rwidth=1.0)
        ax.set_title(f'Histograma da Imagem {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Intensidade de Pixel')
        ax.set_ylabel('Frequência')
        # ax.set_ylim(0, 150000)
        ax.grid(True, alpha=0.3)
        
        # Adicionar informações estatísticas
        stats = f'Min: {imagem_rgb.min()}\nMax: {imagem_rgb.max()}\nMédia: {imagem_rgb.mean():.2f}'
        ax.text(0.98, 0.97, stats, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def expansao_histograma(self, dados, min_val=0, max_val=255):

        dados_flat = np.array(dados, dtype=float)
        data_min = dados_flat.min()
        data_max = dados_flat.max()
        
        if data_min == data_max:
            return np.full_like(dados_flat, (min_val + max_val) / 2, dtype=float)
        
        L = max_val - min_val + 1
        normalized = (dados_flat - data_min) / (data_max - data_min)
        scaled = np.round(normalized * (L - 1)) 
        
        return scaled.reshape(dados.shape)

    def get_activation(self, canal, name):
        x = np.array(canal, dtype=float)
        if name == "relu":
            return np.maximum(0, x)
        if name == "identidade":
            return x 
        raise ValueError(f"Função de ativação desconhecida: {name}")
            

    def print_Canal(self):
        '''
        IMPRESSÃO DE CADA CANAL
        '''
        # Mantem os valores do canal para visualizacao (sem wrap de uint8)
        R = np.clip(self.RSaida, 0, 255).astype(np.uint8)
        G = np.clip(self.GSaida, 0, 255).astype(np.uint8)
        B = np.clip(self.BSaida, 0, 255).astype(np.uint8)

        saida_imgR = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
        saida_imgG = np.zeros_like(saida_imgR)
        saida_imgB = np.zeros_like(saida_imgR)

        saida_imgR[:, :, 0] = R   # vermelho
        saida_imgG[:, :, 1] = G   # verde
        saida_imgB[:, :, 2] = B   # azul

        # Mostrar imagem
        fig, axes = plt.subplots(1, 3, figsize=(12,5))

        axes[0].imshow(saida_imgR)
        axes[0].set_title("Canal R")
        axes[0].axis("off")

        axes[1].imshow(saida_imgG)
        axes[1].set_title("Canal G")
        axes[1].axis("off")

        axes[2].imshow(saida_imgB)
        axes[2].set_title("Canal B")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    def display_images(self, img1, img2, title1="Imagem Inicial", title2="Imagem Final"):
        '''PARA IMAGEM INICIAL X FINAL'''

        if img1 is None and img2 is None:
            print("No images to display")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display first image
        if img1 is not None:
            ax1.imshow(img1)
            ax1.set_title(f"{title1}\n")
            ax1.axis('off')
        else:
            ax1.set_title(f"{title1}\n(No image)")
            ax1.axis('off')
        
        # Display second image
        if img2 is not None:
            ax2.imshow(img2)
            ax2.set_title(f"{title2}\n")
            ax2.axis('off')
        else:
            ax2.set_title(f"{title2}\n(No image)")
            ax2.axis('off')
        
        plt.tight_layout()
        plt.show()


    def print_Imagem_final(self, start_image):
        imagem_rgb = np.stack((self.RSaida, self.GSaida, self.BSaida), axis=2)
        imagem_rgb = imagem_rgb.astype(np.uint8)
        self.display_images(start_image, imagem_rgb)
        return imagem_rgb

        
    def salvar_imagem(self, nome_arquivo="resultado.png", pasta_saida="output"):
        '''
        Salva a imagem RGB de 24 bits no disco usando PIL
        
        Args:
            nome_arquivo: Nome do arquivo a ser salvo (padrão: "resultado.png")
            pasta_saida: Pasta onde a imagem será salva (padrão: "output")
        '''
        
        # Empilha os canais, limita ao intervalo válido e converte para uint8
        imagem_rgb = np.stack((self.RSaida, self.GSaida, self.BSaida), axis=2)
        imagem_uint8 = np.clip(imagem_rgb, 0, 255).astype(np.uint8)
        # Cria a pasta de saída se não existir
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)
        
        # Caminho completo do arquivo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        caminho_completo = os.path.join(script_dir, "output", nome_arquivo)
        
        # Converte o array NumPy para imagem PIL e salva
        imagem_pil = Image.fromarray(imagem_uint8, mode='RGB')
        imagem_pil.save(caminho_completo)
        
        print(f"Imagem salva com sucesso em: {caminho_completo}")
        print(f"Dimensões: {imagem_uint8.shape[0]}x{imagem_uint8.shape[1]}")
        print(f"Modo: RGB (24 bits)")
        
        return caminho_completo

    def salvar_canais_separados(self, nome_base="canal", pasta_saida="output"):
        """
        Salva os três canais RGB separadamente como imagens coloridas
        (canal R em vermelho, canal G em verde, canal B em azul)
        
        Args:
            nome_base: Nome base para os arquivos (padrão: "canal")
            pasta_saida: Pasta onde as imagens serão salvas (padrão: "output")
        """
        # Cria a pasta de saída se não existir
        if not os.path.exists(pasta_saida):
            os.makedirs(pasta_saida)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Para cada canal, criar uma imagem colorida (canal específico ativo)
        canais = [
            (self.RSaida, 'R', 0),  # índice 0 = canal vermelho
            (self.GSaida, 'G', 1),  # índice 1 = canal verde
            (self.BSaida, 'B', 2)   # índice 2 = canal azul
        ]
        
        for canal_data, canal_nome, canal_idx in canais:
            # Normaliza e converte para uint8
            canal_uint8 = np.clip(canal_data, 0, 255).astype(np.uint8)
            
            # Cria imagem colorida com apenas o canal específico ativo
            imagem_colorida = np.zeros((canal_uint8.shape[0], canal_uint8.shape[1], 3), dtype=np.uint8)
            imagem_colorida[:, :, canal_idx] = canal_uint8
            
            # Nome do arquivo
            nome_arquivo = f"{nome_base}_{canal_nome}.png"
            caminho_completo = os.path.join(script_dir, pasta_saida, nome_arquivo)
            
            # Salva a imagem
            imagem_pil = Image.fromarray(imagem_colorida, mode='RGB')
            imagem_pil.save(caminho_completo)
        
        print(f"\nTotal: 3 canais salvos em '{pasta_saida}'")
