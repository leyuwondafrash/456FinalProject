import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Dataset Preparation
data_dir = "./raw-img" #update if necessary
img_height, img_width = 112, 112
batch_size = 16

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=0.3,
    subset='training',
    seed=27
)

validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=0.3,
    subset='validation',
    seed=27
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# Model Definition
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(len(dataset.class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Model Compilation and Summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='.venv/best_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * (10 ** (epoch / 20)))

callbacks = [tensorboard_callback, early_stopping, model_checkpoint, reduce_lr, learning_rate_scheduler]

# Initial Training
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=callbacks)

# Fine-Tuning
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=20, callbacks=callbacks)

# Combine Training History
combined_history = {
    'accuracy': history.history['accuracy'] + fine_tune_history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
    'loss': history.history['loss'] + fine_tune_history.history['loss'],
    'val_loss': history.history['val_loss'] + fine_tune_history.history['val_loss']
}

# Evaluation and Plotting
test_loss, test_accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss: {test_loss}")
print(f"Validation Accuracy: {test_accuracy}")

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(combined_history['accuracy'])
plt.plot(combined_history['val_accuracy'])
plt.title('ResNet50 accuracy')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(combined_history['loss'])
plt.plot(combined_history['val_loss'])
plt.title('ResNet50 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.ylim(0, 4.0)

plt.savefig("ResNet50_performance.jpg")
plt.show()

# Confusion Matrix
val_preds = model.predict(validation_dataset)
val_preds_classes = np.argmax(val_preds, axis=1)
val_trues_classes = np.concatenate([y.numpy() for x, y in validation_set], axis=0).argmax(axis=1)

conf_matrix = confusion_matrix(val_trues_classes, val_preds_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=dataset.class_names, yticklabels=dataset.class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig("ResNet50_Confusion_Matrix.jpg")
plt.show()
