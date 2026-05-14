import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

print(f"TensorFlow: {tf.__version__}")
print(f"TFMOT: {tfmot.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")




"""
Dear user, when we are writing this there are only three people who knows whats going on
Us and god and by now its only god
The code cost me a keyboard that I smashed due to the quantization keep failing
"""


X_train_raw = pd.read_csv('X_train.csv').values.astype(np.float32)
X_test_raw  = pd.read_csv('X_test.csv').values.astype(np.float32)
y_train_raw = pd.read_csv('y_train.csv').values.ravel().astype(int)
y_test_raw  = pd.read_csv('y_test.csv').values.ravel().astype(int)

NUM_FEATURES = X_train_raw.shape[1]          # 46
NUM_CLASSES  = len(np.unique(y_train_raw))   # 5

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_train_raw, y_train_raw, test_size=0.15, random_state=42, stratify=y_train_raw
)

# Reshape for 1D CNN: (samples, features, 1)
X_train = X_train_s.reshape(-1, NUM_FEATURES, 1)
X_val   = X_val_s.reshape(-1, NUM_FEATURES, 1)
X_test  = X_test_raw.reshape(-1, NUM_FEATURES, 1)
y_train = y_train_s
y_val   = y_val_s
y_test  = y_test_raw

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")


# Normal pooling doesn'T  work for CBAM so have to do that same but in another way:
# I added two sublayers to fix the quant issue:
#   - Have no trainable weights
#   - Have proper get_config()



class ReduceMeanLastAxis(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


class ReduceMaxLastAxis(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()

#squeeze block
def channel_attentionSE(x, reduction_ratio=8, name_prefix='ca'):
    channels = x.shape[-1]
    reduced = max(channels // reduction_ratio, 1)
    s = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gavg')(x)
    s = layers.Dense(reduced, activation='relu', use_bias=True,
                     kernel_initializer='he_normal',
                     name=f'{name_prefix}_fc1')(s)
    s = layers.Dense(channels, activation='sigmoid', use_bias=True,
                     kernel_initializer='he_normal',
                     name=f'{name_prefix}_fc2')(s)
    s = layers.Reshape((1, channels), name=f'{name_prefix}_reshape')(s)
    return layers.Multiply(name=f'{name_prefix}_scale')([x, s])

#CBAM
def channel_attention(x, reduction_ratio=8, name_prefix='ca'):
    channels = x.shape[-1]
    reduced = max(channels // reduction_ratio, 1)

    # Shared MLP layers (created once, called twice for weight sharing)
    fc1 = layers.Dense(reduced, activation='relu', use_bias=True,
                       name=f'{name_prefix}_fc1')
    fc2 = layers.Dense(channels, use_bias=True,
                       name=f'{name_prefix}_fc2')

    avg_pool = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gavg')(x)
    avg_out = fc2(fc1(avg_pool))

    max_pool = layers.GlobalMaxPooling1D(name=f'{name_prefix}_gmax')(x)
    max_out = fc2(fc1(max_pool))

    combined = layers.Add(name=f'{name_prefix}_add')([avg_out, max_out])
    attention = layers.Activation('sigmoid', name=f'{name_prefix}_sig')(combined)
    attention = layers.Reshape((1, channels), name=f'{name_prefix}_reshape')(attention)

    return layers.Multiply(name=f'{name_prefix}_scale')([x, attention])


def spatial_attention(x, kernel_size=7, name_prefix='sa'):

    avg_pool = ReduceMeanLastAxis(name=f'{name_prefix}_avg')(x)  # (batch, steps, 1)
    max_pool = ReduceMaxLastAxis(name=f'{name_prefix}_max')(x)   # (batch, steps, 1)

    concat = layers.Concatenate(axis=-1, name=f'{name_prefix}_cat')(
        [avg_pool, max_pool]
    )  # (batch, steps, 2)

    attention = layers.Conv1D(
        filters=1, kernel_size=kernel_size, padding='same',
        activation='sigmoid', use_bias=False,
        kernel_initializer='he_normal',
        name=f'{name_prefix}_conv'
    )(concat)  # (batch, steps, 1)

    return layers.Multiply(name=f'{name_prefix}_scale')([x, attention])


def cbam_block(x, reduction_ratio=8, kernel_size=7, name_prefix='cbam'):
    """Full CBAM: Channel Attention -> Spatial Attention."""
    x = channel_attention(x, reduction_ratio, name_prefix=f'{name_prefix}_ch')
    x = spatial_attention(x, kernel_size, name_prefix=f'{name_prefix}_sp')
    return x




# 3. Build CNN-CBAM Model

def build_cnn_cbam(input_shape, num_classes, cbam_reduction, cbam_kernel):
    inputs = layers.Input(shape=input_shape, name='input')

    x = layers.Conv1D(8, kernel_size=3, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = cbam_block(x, reduction_ratio=cbam_reduction,
                   kernel_size=cbam_kernel, name_prefix='cbam1')
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='pool1')(x)

    x = layers.Conv1D(8, kernel_size=3, padding='same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = cbam_block(x, reduction_ratio=cbam_reduction,
                   kernel_size=cbam_kernel, name_prefix='cbam2')
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='pool2')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(16, name='fc')(x)
    x = layers.Activation('relu', name='fc_relu')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inputs, outputs, name='CNN_CBAM')

#tog efter cbam pappret
model = build_cnn_cbam(
    input_shape=(NUM_FEATURES, 1),
    num_classes=NUM_CLASSES,
    cbam_reduction=8,
    cbam_kernel=7,
)
model.summary()


# Train Float32 Baseline

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Val')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Val')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_f32.png', dpi=150)
plt.show()

# Float32 evaluation
y_pred_f32 = model.predict(X_test, verbose=0).argmax(axis=1)

print("=" * 60)
print("FLOAT32 BASELINE")
print("=" * 60)
print(classification_report(y_test, y_pred_f32, target_names=CLASS_NAMES, digits=4))

baseline_acc = np.mean(y_pred_f32 == y_test)
print(f"Test Accuracy: {baseline_acc:.4f}")


from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import (
    Default8BitQuantizeConfig,
)
conv1d_qconfig = Default8BitQuantizeConfig(['kernel'], ['activation'], False)


def apply_quantization(layer):
    if isinstance(layer, keras.layers.Conv1D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, conv1d_qconfig)
    if isinstance(layer, keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer


custom_objects = {
    'ReduceMeanLastAxis': ReduceMeanLastAxis,
    'ReduceMaxLastAxis': ReduceMaxLastAxis,
}

# Clone
with keras.utils.custom_object_scope(custom_objects):
    annotated_model = keras.models.clone_model(
        model, clone_function=apply_quantization
    )

# quantize_scope 
#Pray
with tfmot.quantization.keras.quantize_scope(custom_objects):
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

qat_model.summary()

# lower LR 
qat_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

qat_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
]

qat_history = qat_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    callbacks=qat_callbacks,
    verbose=1
)

# QAT training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(qat_history.history['loss'], label='Train')
ax1.plot(qat_history.history['val_loss'], label='Val')
ax1.set_title('QAT Fine-tuning: Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(qat_history.history['accuracy'], label='Train')
ax2.plot(qat_history.history['val_accuracy'], label='Val')
ax2.set_title('QAT Fine-tuning: Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_qat.png', dpi=150)
plt.show()


# to TFLite INT8
# keeps fucking crashing piece of shit
def convert_to_tflite_int8(qat_model, representative_data, output_path='cnn_SE_int8.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"INT8 TFLite model saved: {output_path}")
    print(f"Model size: {size_kb:.1f} KB")
    return output_path, size_kb


def representative_dataset_gen():
    indices = np.random.choice(len(X_train), size=min(1000, len(X_train)), replace=False)
    for i in indices:
        yield [X_train[i:i+1]]


tflite_path, model_size_kb = convert_to_tflite_int8(
    qat_model,
    representative_dataset_gen,
    output_path='cnn_cbam_qat_int8_NEW.tflite'
)


#INT8 Evaluation

class TFLitePredictor:

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.input_scale = self.input_details['quantization'][0]
        self.input_zp = self.input_details['quantization'][1]
        self.output_scale = self.output_details['quantization'][0]
        self.output_zp = self.output_details['quantization'][1]

        print(f"Input:  dtype={self.input_details['dtype']}, "
              f"shape={self.input_details['shape']}, "
              f"scale={self.input_scale:.6f}, zp={self.input_zp}")
        print(f"Output: dtype={self.output_details['dtype']}, "
              f"shape={self.output_details['shape']}, "
              f"scale={self.output_scale:.6f}, zp={self.output_zp}")

    def predict_one(self, x_float):
        """Run inference on a single float32 sample."""
        x_int8 = np.round(x_float / self.input_scale + self.input_zp).astype(np.int8)
        self.interpreter.set_tensor(self.input_details['index'], x_int8)
        self.interpreter.invoke()
        output_int8 = self.interpreter.get_tensor(self.output_details['index'])
        output_float = (output_int8.astype(np.float32) - self.output_zp) * self.output_scale
        return output_float

    def predict_batch(self, X_float):
        """Run inference on a batch (sample-by-sample for TFLite)."""
        preds = []
        for i in range(len(X_float)):
            out = self.predict_one(X_float[i:i+1])
            preds.append(out.argmax(axis=1)[0])
        return np.array(preds)


tflite_pred = TFLitePredictor(tflite_path)

y_pred_int8 = tflite_pred.predict_batch(X_test)

print("=" * 60)
print("INT8 QAT MODEL")
print("=" * 60)
print(classification_report(y_test, y_pred_int8, target_names=CLASS_NAMES, digits=4))

int8_acc = np.mean(y_pred_int8 == y_test)
print(f"Test Accuracy: {int8_acc:.4f}")
print(f"Accuracy drop vs FP32: {(baseline_acc - int8_acc) * 100:.2f}%")


#Benchmarking

def benchmark_latency(predictor, X_sample, n_runs=500, warmup=50):
    """Measure per-sample inference latency."""
    sample = X_sample[0:1]
    for _ in range(warmup):
        predictor.predict_one(sample)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predictor.predict_one(sample)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        'mean_ms':   np.mean(latencies),
        'median_ms': np.median(latencies),
        'p95_ms':    np.percentile(latencies, 95),
        'p99_ms':    np.percentile(latencies, 99),
        'std_ms':    np.std(latencies)
    }


int8_latency = benchmark_latency(tflite_pred, X_test)


class KerasPredictor:
    def __init__(self, model):
        self.model = model

    def predict_one(self, x):
        return self.model(x, training=False).numpy()


f32_latency = benchmark_latency(KerasPredictor(model), X_test)

model.save('cnn_cbam_f32.keras')
f32_size_kb = os.path.getsize('cnn_cbam_f32.keras') / 1024

print("\n" + "=" * 60)
print("BENCHMARK COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'Float32':>12} {'INT8 QAT':>12} {'Ratio':>10}")
print("-" * 60)
print(f"{'Accuracy':<25} {baseline_acc:>11.4f} {int8_acc:>11.4f} {int8_acc/baseline_acc:>9.4f}x")
print(f"{'Model Size (KB)':<25} {f32_size_kb:>11.1f} {model_size_kb:>11.1f} {f32_size_kb/model_size_kb:>9.1f}x")
print(f"{'Latency Mean (ms)':<25} {f32_latency['mean_ms']:>11.3f} {int8_latency['mean_ms']:>11.3f} {f32_latency['mean_ms']/int8_latency['mean_ms']:>9.1f}x")
print(f"{'Latency P95 (ms)':<25} {f32_latency['p95_ms']:>11.3f} {int8_latency['p95_ms']:>11.3f} {f32_latency['p95_ms']/int8_latency['p95_ms']:>9.1f}x")
print(f"\nINT8 model fits in 256KB RAM: {'YES' if model_size_kb < 256 else 'NO'} ({model_size_kb:.1f} KB)")
print(f"INT8 latency < 10ms:          {'YES' if int8_latency['mean_ms'] < 10 else 'NO'} ({int8_latency['mean_ms']:.3f} ms)")

