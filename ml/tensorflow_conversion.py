"""Optional OFFICIAL TensorFlow -> TensorFlow Lite converter.

The included model was trained in PyTorch then serialized in supported TFLite
FlatBuffer schema by train_export.py because TensorFlow is not installed in the
build environment. This script reproduces its architecture in TensorFlow/Keras
and converts the saved trained weights with tf.lite.TFLiteConverter.

Requirements: pip install tensorflow numpy
Usage from project root: python ml/tensorflow_conversion.py
Writes a separately named .tflite; compare outputs against the app's bundled
model before replacing the app model.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
w=np.load(ROOT/'ml/weights.npz')
model=tf.keras.Sequential([
    tf.keras.Input(shape=(1024,),dtype=tf.float32),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Softmax(),
])
model.layers[0].set_weights([w['w0'].T, w['b0']])
# Fold Softmax beta=0.25 into the preceding dense layer to match the packaged model.
model.layers[1].set_weights([w['w1'].T*0.25, w['b1']*0.25])
converter=tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[]
tflite_bytes=converter.convert()
output=ROOT/'ml/sketchai_tensorflow_converted.tflite'
output.write_bytes(tflite_bytes)
print('Saved official TensorFlow conversion:',output)
print('Inputs:',model.input_shape,'Outputs:',model.output_shape)
print('This conversion is an optional reproducibility route, not yet run in this environment.')
