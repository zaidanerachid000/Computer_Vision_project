
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

# Chargement des données
train_data = load_data('../../../../CV_dataset/train/train')

# Extraction des données
X = train_data[b'data']          
y = train_data[b'fine_labels']   

# --- TRAITEMENT D'UNE IMAGE ---
img = X[79]

# Reshape pour passer de 1D (3072) à 3D (Channels, Height, Width)
# CIFAR est stocké en (3, 32, 32)
img = img.reshape(3, 32, 32)


# Transpose pour mettre les canaux à la fin (Height, Width, Channels)
# C'est le format attendu par Matplotlib
img = np.transpose(img, (1, 2, 0))

# Conversion en uint8 pour éviter les problèmes d'affichage avec imshow
img = img.astype("uint16")

# --- AFFICHAGE ---
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(f"Label: {y[79]}")
plt.axis('off') # Cache les axes
plt.show()

# Affichage avec interpolation bilinéaire
plt.imshow(img, interpolation='bilinear')
plt.axis('off')
plt.show()