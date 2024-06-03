import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the path to your dataset directories
train_dir = './dataset/train'
test_dir = './dataset/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the datasets with data augmentation
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(32, 32),
    batch_size=32,
    label_mode='categorical'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=32,
    label_mode='categorical'
)

# Define hyperparameters
HYPERPARAMETERS = {
    'num_filters': [16, 32, 64],
    'kernel_size': [3, 5],
    'learning_rate': [1e-4, 1e-5],  # Reduced learning rate
    'dropout_rate': [0.4, 0.5],  # Increased dropout rate
    'l2_regularization': [0.1, 0.2]  # Increased L2 regularization strength
}

# Define the model building function
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Choice('num_filters', HYPERPARAMETERS['num_filters']),
            kernel_size=hp.Choice('kernel_size', HYPERPARAMETERS['kernel_size']),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(hp.Choice('l2_regularization', HYPERPARAMETERS['l2_regularization'])),
            input_shape=(32, 32, 3)
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(hp.Choice('dropout_rate', HYPERPARAMETERS['dropout_rate'])),
        keras.layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', HYPERPARAMETERS['learning_rate'])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparameters',
    project_name='wand_casting'
)

# Perform hyperparameter tuning
tuner.search(train_dataset,
             validation_data=validation_dataset,
             epochs=10)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(train_dataset, validation_data=validation_dataset, epochs=100)

# Evaluate the trained model
test_loss, test_acc = best_model.evaluate(test_dataset)
print('\nTest accuracy:', test_acc)

# Save the model
best_model.save('wand_casting_classifier_tuned.h5')

# Optionally, plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()