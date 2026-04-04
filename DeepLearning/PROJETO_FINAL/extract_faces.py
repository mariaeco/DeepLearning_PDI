from mtcnn import MTCNN
from PIL import Image
from os import listdir, rename, makedirs
from os.path import isdir, isfile, join, splitext
from numpy import asarray


class ProcessImages:
    def __init__(self) -> None:
        self.detector = MTCNN()

    def extrair_face(self, file, size=(160,160)):
        img = Image.open(file)
        img = img.convert('RGB')
        array_img = asarray(img)
        results = self.detector.detect_faces(array_img)
        if not results:
            return None

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1+width, y1+height
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
            ''' Rename fotos '''

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
            # self.rename_imgs(path, subdir)
            self.load_imgs(path, path_target)

            


if __name__ == "__main__":
    img_processing = ProcessImages()
    # img_processing.load_dir('imgs/','faces/')
    img_processing.load_dir('imagens_teste/','faces_validation/')