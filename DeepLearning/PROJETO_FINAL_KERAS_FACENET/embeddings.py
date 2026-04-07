from __future__ import annotations
import warnings
from os import environ, listdir, makedirs, rename

# Antes de carregar TensorFlow (MTCNN) e face_recognition: reduz ruído no terminal.
environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
environ.setdefault("GRPC_VERBOSITY", "ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")


from PIL import Image
from os import listdir
from os.path import isdir, join
from numpy import asarray, expand_dims
import numpy as np
import pandas as pd

from keras_facenet import FaceNet




def load_face(filename):
        image = Image.open(filename)
        image = image.convert("RGB")
        return asarray(image)


def carregar_faces(dir_faces):
    faces = list()
    for filename in listdir(dir_faces):
        path = join(dir_faces, filename)
        try:
            faces.append(load_face(path))
        except:
            print("Erro na imagem {}".format(path))
    
    return faces

#Carregar todo o dataset
def load_dir(directory_src):
    X, y = list(), list()
    
    for subdir in listdir(directory_src):
        path = join(directory_src,subdir)
        
        if not isdir(path):
                continue
            
        faces = carregar_faces(path)
        labels = [subdir for _ in range(len(faces))]

        print('>Carregadas %d faces da class %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)



if __name__ == "__main__":
    trainX, trainy= load_dir(directory_src='faces/')

    embedder = FaceNet()
    embeddings = embedder.embeddings(trainX)

    df = pd.DataFrame(data=embeddings)
    df['target'] = trainy
    df.to_csv('faces.csv', index=False)
    
    
    #Para validação - usar em main.py (acuracia dos modelos)
    # valX, valy= load_dir(directory_src='faces_validation/')
    # embedder = FaceNet()
    # embeddingsVal = embedder.embeddings(valX)
    # df_val = pd.DataFrame(data=embeddingsVal)
    # df_val['target'] = valy
    # df_val.to_csv('faces_validation.csv', index=False)
    