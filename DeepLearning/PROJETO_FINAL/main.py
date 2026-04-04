#Ordem de rodagem: 1) extract_faces.py, 2) embeddings.py, 3) main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

#modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import svm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

df = pd.read_csv('faces.csv')
X = np.array(df.drop(columns=['target', 'Unnamed: 0'], errors='ignore'))
y = np.array(df.target)


trainX, trainY = shuffle(X,y, random_state = 0)
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)

df_val = pd.read_csv('faces_validation.csv')
valX = np.array(df_val.drop(columns=['target', 'Unnamed: 0'], errors='ignore'))
valY = np.array(df_val.target)
valY = out_encoder.transform(valY)

#KNN ----------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(trainX, trainY)
yhat_train = knn.predict(trainX)
yhat_val = knn.predict(valX)
acc_train = accuracy_score(trainY, yhat_train)
acc_val = accuracy_score(valY, yhat_val)

print('------------- KNN -------------')
print(f"Acuracia treino: {acc_train:.4f}")
print(f"Acuracia validacao: {acc_val:.4f}")


#matrix de confusao
cm = confusion_matrix(valY, yhat_val)
cm_df = pd.DataFrame(
    cm,
    index=out_encoder.classes_,
    columns=out_encoder.classes_
)

# plot da matriz de confusao
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=out_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusao - KNN')
plt.tight_layout()
plt.show()



#SVM ----------------------------------------------------------
svm = svm.SVC()
svm.fit(trainX, trainY)

yhat_train = svm.predict(trainX)
yhat_val = svm.predict(valX)
acc_train = accuracy_score(trainY, yhat_train)
acc_val = accuracy_score(valY, yhat_val)

print('------------- SVM -------------')
print(f"Acuracia treino: {acc_train:.4f}")
print(f"Acuracia validacao: {acc_val:.4f}")

# plot da matriz de confusao
cm_svm = confusion_matrix(valY, yhat_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=out_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusao - SVM')
plt.tight_layout()
plt.show()



#KERAS ----------------------------------------------------------
num_classes = len(out_encoder.classes_)
trainY_cat = to_categorical(trainY, num_classes=num_classes)
valY_cat = to_categorical(valY, num_classes=num_classes)

model = models.Sequential([
    layers.Input(shape=(trainX.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(trainX, trainY_cat, epochs=100, batch_size=8, verbose=0)


yhat_train = model.predict(trainX, verbose=0)
yhat_val = model.predict(valX, verbose=0)
yhat_train = np.argmax(yhat_train, axis=1)
yhat_val = np.argmax(yhat_val, axis=1)

acc_train = accuracy_score(trainY, yhat_train)
acc_val = accuracy_score(valY, yhat_val)

print('------------- KERAS -------------')

print(f"Acuracia treino: {acc_train:.4f}")
print(f"Acuracia validacao: {acc_val:.4f}")


# plot da matriz de confusao
cm_svm = confusion_matrix(valY, yhat_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=out_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusao - KERAS')
plt.tight_layout()
plt.show()

