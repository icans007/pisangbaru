import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Konfigurasi Utama
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3 

TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'

# Augmentasi Data menggunakan Layer Internal TensorFlow (Sangat Stabil)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.07), # Sekitar 25 derajat
    tf.keras.layers.RandomZoom(0.2),
])

def build_model(base_model_name='vgg16'):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Terapkan augmentasi hanya saat training
    x = data_augmentation(inputs)
    
    if base_model_name == 'vgg16':
        # VGG16 Preprocessing Layer (Rescale ke 1./255)
        x = Lambda(lambda img: img / 255.0)(x)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        x = base_model(x)
    else:
        # ResNet50 Preprocessing Layer (Konversi otomatis RGB ke BGR + Mean Subtraction)
        x = Lambda(lambda img: preprocess_resnet(img))(x)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:-6]:
            layer.trainable = False
        x = base_model(x)
        
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) 
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    # Memuat Dataset Menggunakan Utilitas Modern (Format Alfabetis Otomatis)
    print("\n--- MEMUAT DATASET PISANG ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    # Cetak kepastian urutan indeks kelas
    print("\nUrutan indeks kelas berdasarkan urutan folder alfabetis:")
    class_names = train_ds.class_names
    for index, name in enumerate(class_names):
        print(f"Indeks {index} : {name}")
    print("-" * 50)

    # -------------------------------------------------------------------------
    # 1. TRAINING VGG16
    # -------------------------------------------------------------------------
    print("\n--- [1/2] TRAINING MODEL VGG16 ---")
    vgg_model = build_model('vgg16')
    vgg_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    vgg_model.save('models/vgg16_banana.h5')
    print("Model VGG16 Berhasil Disimpan!")

    # -------------------------------------------------------------------------
    # 2. TRAINING RESNET50
    # -------------------------------------------------------------------------
    print("\n--- [2/2] TRAINING MODEL RESNET50 ---")
    resnet_model = build_model('resnet50')
    resnet_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    resnet_model.save('models/resnet50_banana.h5')
    
    print("\n[SUKSES] Semua model selesai di-training menggunakan koordinat data yang sinkron!")

if __name__ == '__main__':
    main()