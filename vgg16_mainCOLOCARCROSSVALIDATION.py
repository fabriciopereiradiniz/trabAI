import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

output_dir = 'confusion_matrices'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

species_dirs = {
    'Sicalis_flaveola': glob.glob(os.path.join('species/Sicalis_flaveola/', '*')),
    'Sporophila_maximiliani': glob.glob(os.path.join('species/Sporophila_maximiliani/', '*')),
    'Zenaida_auriculata': glob.glob(os.path.join('species/Zenaida_auriculata/', '*')),
    'Sporophila_angolensis': glob.glob(os.path.join('species/Sporophila_angolensis/', '*')),
    'Sporophila_caerulescens': glob.glob(os.path.join('species/Sporophila_caerulescens/', '*')),
    'Paroaria_dominicana': glob.glob(os.path.join('species/Paroaria_dominicana/', '*')),
    'Saltator_similis': glob.glob(os.path.join('species/Saltator_similis/', '*')),
    'Sporophila_albogularis': glob.glob(os.path.join('species/Sporophila_albogularis/', '*')),
    'Sporophila_nigricollis': glob.glob(os.path.join('species/Sporophila_nigricollis/', '*')),
    'Sporophila_lineola': glob.glob(os.path.join('species/Sporophila_lineola/', '*'))
}

X = []
y = []

category_map = {species: idx for idx, species in enumerate(species_dirs.keys())}

image_counts = {}

for species, paths in species_dirs.items():
    species_X = []
    species_y = []
    
    for f in paths:
        try:
            img = cv.imread(f)
            if img is None:
                print(f"Erro: Não foi possível carregar a imagem {f}")
                continue
            img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
            X.append(img)
            y.append(category_map[species])
            species_X.append(img)
            species_y.append(category_map[species])
        except Exception as e:
            print(f"Erro ao processar {f}: {e}")
    
    image_counts[species] = len(species_X)

X = np.array(X)
y = np.array(y)

X = X / 255.0

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

datagen = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

accuracy_scores = []
f1_scores = []

for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    datagen.fit(X_train)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  

    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=10,
        validation_data=(X_val, y_val)
    )
    
    y_val_pred = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracy_scores.append(accuracy)
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    f1_scores.append(f1)
    
    print(f"Acurácia da validação nesta rodada: {accuracy:.4f}")
    print(f"F-score da validação nesta rodada: {f1:.4f}")
    
    cm = confusion_matrix(y_val, y_val_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=category_map.keys(), yticklabels=category_map.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Fold {len(accuracy_scores)}')
    
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{len(accuracy_scores)}.png'))
    plt.close()

mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Acurácia média após {skf.get_n_splits()} folds: {mean_accuracy:.4f}")
print(f"Desvio padrão das acurácias: {std_accuracy:.4f}")
print(f"F-score médio após {skf.get_n_splits()} folds: {mean_f1:.4f}")
print(f"Desvio padrão dos F-scores: {std_f1:.4f}")

model.save('vgg16_trained_model_kfold.h5')
