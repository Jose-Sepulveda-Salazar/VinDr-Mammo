# -*- coding: utf-8 -*-

import tensorflow as tf
import nni
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Obtener hiperparametros desde NNI
params = nni.get_next_parameter()

# Directorios principales del dataset
train_dir =  '/home/semillerolun/Jose_documentos/Dataset128x128_2/train'
test_dir =  '/home/semillerolun/Jose_documentos/Dataset128x128_2/test'

# Generador de datos 
datagen = ImageDataGenerator()

# Tamaño del lote
batch_size = params['batch_size']

# Generador para el conjunto de entrenamiento
train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb', # Convertir automáticamente de 1 canal a 3 canales iguales
    shuffle=True,
    seed=42
)

test_generator = datagen.flow_from_directory(
    directory=test_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    seed=42
)

# Definir una función de ajuste de la tasa de aprendizaje
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9  # Reducir la tasa de aprendizaje en un 10% cada época después de la 9

# Callbacks
lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=7, 
                               restore_best_weights=True,
                               start_from_epoch=20)


# Definimos una funcion para crear el modelo base utilizando EfficientNetB0
def create_base_model(input_shape):
    # Construccion del modelo utilizando EfficientNetB0
    effnet = EfficientNetB0(weights='imagenet', 
                            include_top=False, 
                            input_shape=input_shape)
    
    # Congelar todas las capas
    for layer in effnet.layers:
        layer.trainable = False

    # Descongelar las ultimas `unfreeze_layers` capas
    if params['unfreeze_layers'] > 0:
        for layer in effnet.layers[-params['unfreeze_layers']:]:
            layer.trainable = True   
    
    # Retornar el modelo base con las capas convolucionales de EfficientNet
    return Model(inputs=effnet.input, outputs=effnet.output)

# Tamano de la imagen de entrada
input_shape = (128, 128, 3)

# Crear modelos base 
base_model = create_base_model(input_shape)

# Construcción del modelo con capas densas dinámicas
input = Input(shape=input_shape, name='input')
x = base_model(input)
x = GlobalAveragePooling2D()(x) # Aplicamos un promedio global en las salidas del modelo base

#Agregar capas densas de acuerdo al número dinámico definido por `num_dense_layers`
num_dense_layers = params['num_dense_layers']
# Aplicamos un promedio global en las salidas del modelo base
for i in range(1, num_dense_layers + 1):
    # Redondear el valor de dropout a 4 y 6 decimales
    dropout_value = round(params[f'dropout_{i}'], 4)
    l1_regularization = round(params[f'L1_regularization_{i}'], 6)
    l2_regularization = round(params[f'L2_regularization_{i}'], 6)

    x = Dropout(dropout_value, name=f'dropout_{i}')(x)
    x = Dense(params[f'dense_{i}'], activation='relu', kernel_regularizer=L1L2(l1_regularization, l2_regularization))(x)

# Capa de salida
output = Dense(2, activation='softmax', name='output')(x)

# Redondear el valor de learning_rate a 6 decimales
learning_rate = round(params['learning_rate'], 6)

# Crear el modelo final
model = Model(inputs=input, outputs=output)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    
# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(train_generator,
                    epochs=params['epochs'],
                    validation_data=test_generator,
                    verbose=1,
                    callbacks=[lr_scheduler, early_stopping])
                    
# Obtener el mejor valor de precision de entrenamiento
best_test_acc = max(history.history['val_accuracy'])

# Reportar el resultado final a NNI
nni.report_final_result(best_test_acc)