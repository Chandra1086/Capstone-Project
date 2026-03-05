import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os

# ===============================
# CONFIG
# ===============================
DATASET_PATH = "DataSet"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
MODEL_NAME = "traffic_hand_signal_cnn.h5"

# ===============================
# DATA GENERATORS (STRONG AUGMENTATION)
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.6, 1.4],
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_gen.num_classes
print("Number of classes:", NUM_CLASSES)
print("Class indices:", train_gen.class_indices)

# ===============================
# MOBILENETV2 BASE MODEL
# ===============================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model (VERY IMPORTANT)
base_model.trainable = False

# ===============================
# CUSTOM CLASSIFIER HEAD
# ===============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ===============================
# COMPILE
# ===============================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# EARLY STOPPING
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ===============================
# TRAIN
# ===============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ===============================
# SAVE MODEL
# ===============================
model.save(MODEL_NAME)
print(f"✅ Model saved as {MODEL_NAME}")
