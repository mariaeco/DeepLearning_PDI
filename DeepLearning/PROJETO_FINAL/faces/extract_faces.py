from mtcnn import MTCNN
from PIL import Image
from os.path import listdir
from os.path import isdir
from numpy import asarray



detector = MTCNN()

def extrair_face(file, size=(160,160)):
    img = Image.open(file)
    img = img.convert('RGB')
    array_img = asarray(img)
    