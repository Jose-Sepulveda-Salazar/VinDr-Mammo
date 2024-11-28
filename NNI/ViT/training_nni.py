import tensorflow_hub as hub
import nni
import os  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import preprocess_input

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Obtener hiperparámetros desde NNI
params = nni.get_next_parameter()

# Definir los hiperparámetros con valores por defecto en caso de que NNI no los proporcione
epochs = params.get('epochs', 30)
learning_rate = params.get('learning_rate', 0.00031448)
dropout_1 = params.get('dropout_1', 0.3)
dropout_2 = params.get('dropout_2', 0.35)
dropout_3 = params.get('dropout_3', 0.4)
dropout_4 = params.get('dropout_4', 0.4)
dropout_5 = params.get('dropout_5', 0.4)
dropout_6 = params.get('dropout_6', 0.4)
dropout_7 = params.get('dropout_7', 0.3)
dense_1 = params.get('dense_1', 128)
dense_2 = params.get('dense_2', 192)
dense_3 = params.get('dense_3', 192)
dense_4 = params.get('dense_4', 256)
dense_5 = params.get('dense_5', 128)
dense_6 = params.get('dense_6', 64)
l2_regularization_1 = params.get('l2_regularization_1', 0.005)
l2_regularization_2 = params.get('l2_regularization_2', 0.004)
l2_regularization_3 = params.get('l2_regularization_3', 0.0005)
l2_regularization_4 = params.get('l2_regularization_4', 0.003)
l2_regularization_5 = params.get('l2_regularization_5', 0.003)
l2_regularization_6 = params.get('l2_regularization_6', 0.002)

# Definir los generadores de datos
train_dir =  'C:/Users/jsepu/OneDrive/Documentos/Trabajo_redes/Mamografia/Dataset/train'
test_dir =   'C:/Users/jsepu/OneDrive/Documentos/Trabajo_redes/Mamografia/Dataset/test'

# Generadores de datos con ImageDataGenerator
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Generador para train
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb', # Convertir automáticamente de 1 canal a 3 canales iguales
    shuffle=False,
    seed=42)

# Generador para test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb', 
    shuffle=False,
    seed=42)

# Cálculo de steps_per_epoch y validation_steps
steps_per_epoch = len(train_generator)
validation_steps = len(test_generator)

# Definimos una función para crear el modelo base utilizando Vision Transformer (ViT)
def create_base_model(input_shape):
    # Cargar el modelo ViT desde TensorFlow Hub
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"  # Vision Transformer base 16
    vit_layer = hub.KerasLayer(vit_url, trainable=False)  # Congelar todas sus capas
    
    # Definir el input y el modelo
    inputs = Input(shape=input_shape)
    x = vit_layer(inputs)  # Usamos ViT como extractor de características
    
    return Model(inputs=inputs, outputs=x)

# Tamaño de la imagen de entrada (ViT espera imágenes de 224x224)
input_shape = (224, 224, 3)

# Crear modelos base 
base_model = create_base_model(input_shape)

# Definir la entrada y salida del modelo
input = Input(shape=input_shape, name='input')
x = base_model(input)

# Capas densas
x = Dropout(dropout_1, name='dropout_1')(x)
x = Dense(dense_1, activation='relu', kernel_regularizer=l2(l2_regularization_1), name='dense_1')(x)
x = BatchNormalization(name='batch_norm_1')(x)

x = Dropout(dropout_2, name='dropout_2')(x)
x = Dense(dense_2, activation='relu', kernel_regularizer=l2(l2_regularization_2), name='dense_2')(x)
x = BatchNormalization(name='batch_norm_2')(x)

x = Dropout(dropout_3, name='dropout_3')(x)
x = Dense(dense_3, activation='relu', kernel_regularizer=l2(l2_regularization_3), name='dense_3')(x)
x = BatchNormalization(name='batch_norm_3')(x)

x = Dropout(dropout_4, name='dropout_4')(x)
x = Dense(dense_4, activation='relu', kernel_regularizer=l2(l2_regularization_4), name='dense_4')(x)
x = BatchNormalization(name='batch_norm_4')(x)

x = Dropout(dropout_5, name='dropout_5')(x)
x = Dense(dense_5, activation='relu', kernel_regularizer=l2(l2_regularization_5), name='dense_5')(x)
x = BatchNormalization(name='batch_norm_5')(x)

x = Dropout(dropout_6, name='dropout_6')(x)
x = Dense(dense_6, activation='relu', kernel_regularizer=l2(l2_regularization_6), name='dense_6')(x)
x = BatchNormalization(name='batch_norm_6')(x)

# Capa de salida
output = Dropout(dropout_7, name='dropout_7')(x)
output = Dense(2, activation='softmax')(output)  # 2 etiquetas de clasificación (BI-RADS)

# Definir el modelo con una sola entrada
model = Model(inputs=input, outputs=output)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=validation_steps
                    )

# Obtener el mejor valor de precisión de validación
best_train_acc = max(history.history['accuracy'])

# Reportar el resultado final a NNI
nni.report_final_result(best_train_acc)