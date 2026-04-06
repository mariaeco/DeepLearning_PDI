from __future__ import annotations


from mtcnn import MTCNN
from PIL import Image
from os import listdir, rename, makedirs
from os.path import isdir, isfile, join, splitext
from numpy import asarray

import warnings
from os import environ, listdir, makedirs, rename

# Antes de carregar TensorFlow (MTCNN) e face_recognition: reduz ruído no terminal.
environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
environ.setdefault("GRPC_VERBOSITY", "ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")


class ProcessImages:
    def __init__(self) -> None:
        self.detector = MTCNN()

    def extrair_face(self, file, size=(250, 250)):
        img = Image.open(file)
        img = img.convert("RGB")
        array_img = asarray(img)
        results = self.detector.detect_faces(array_img)
        if not results:
            return None

        x1, y1, width, height = results[0]["box"]
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

        face = array_img[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize(size)
        
        return image


    def flip_image(self, image):
        if image is None:
            return None
        img = image.transpose(Image.FLIP_LEFT_RIGHT)
        return img


    def load_imgs(self, dir_source, dir_target):
        print(dir_source, dir_target)
        makedirs(dir_target, exist_ok=True)
        for filename in listdir(dir_source):
            path = join(dir_source, filename)
            path_tg = join(dir_target, filename)
            flip_name = 'flip' + filename
            path_flip = join(dir_target, flip_name)

            if not isfile(path):
                continue

            print(path,'-->', path_tg)
            face = self.extrair_face(path)
            if face is None:
                print(f"Rosto nao detectado em: {filename}")
                continue
            flip = self.flip_image(face)

            face.save(path_tg, "JPEG",quality=100, optimize=True)
            flip.save(path_flip, "JPEG",quality=100, optimize=True)
            


    def rename_imgs(self, path,subdir):
            ''' Renomear fotos se necessario (ativar em load_dir)'''

            files = [
                file_name for file_name in sorted(listdir(path))
                if isfile(join(path, file_name))
            ]

            print(f"Subdiretorio: {subdir} ({len(files)} arquivos)")
            for old_name in files:
                old_file = join(path, old_name)
                _, ext = splitext(old_name)
                new_name = f"{self.global_index:06d}{ext.lower()}"
                new_file = join(path, new_name)

                rename(old_file, new_file)
                # print(f"  {old_name} -> {new_name}")
                self.global_index += 1


    def load_dir(self, dir_source, dir_target):
        self.global_index = 1
        for subdir in sorted(listdir(dir_source)):
            path = join(dir_source, subdir)
            path_target = join(dir_target, subdir)

            if not isdir(path):
                continue

            #to rename fotos if needed
            self.rename_imgs(path, subdir)
            self.load_imgs(path, path_target)

            


if __name__ == "__main__":
    img_processing = ProcessImages()
    img_processing.load_dir('imgs/','faces/')

    #Para validação - usar em main.py (acuracia dos modelos)
    #mas aqui é acuracia dos embeddings do keras_facenet e não por meu treinamento
    img_processing.load_dir('imgs_validation/','faces_validation/')