
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow: {tf.__version__}")
print(f"TFMOT: {tfmot.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")


#Början sammas om andra

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


# SeparableConv1D instead of Conv1D. 
def build_ds_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name='input')

# Block 1
    x = layers.SeparableConv1D(
        filters=8, kernel_size=3, padding='same', use_bias=False,
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        name='sepconv1'
    )(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='pool1')(x)

# Block 2: 
    x = layers.SeparableConv1D(
        filters=8, kernel_size=3, padding='same', use_bias=False,
        depthwise_initializer='he_normal',
        pointwise_initializer='he_normal',
        name='sepconv2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='pool2')(x)

    # Classifier head (same as CNN-CBAM)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(16, name='fc')(x)
    x = layers.Activation('relu', name='fc_relu')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return Model(inputs, outputs, name='DS_CNN')


model = build_ds_cnn(
    input_shape=(NUM_FEATURES, 1),
    num_classes=NUM_CLASSES,
)
model.summary()


#Train Float32 Baseline

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

sep_conv_qconfig = Default8BitQuantizeConfig(
    ['depthwise_kernel', 'pointwise_kernel'], ['activation'], False
)
 
 



#Yoinked the QAT from the CBAM, changed some stuff but works now /anton jew jew jew jew

def apply_quantization(layer):
    if isinstance(layer, keras.layers.SeparableConv1D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, sep_conv_qconfig)
    if isinstance(layer, keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer
 
 
annotated_model = keras.models.clone_model(model, clone_function=apply_quantization)
 
qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
qat_model.summary()
 
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

 
# INT8 QAT
 
def convert_to_tflite_int8(keras_model, representative_data, output_path='ds_cnn_int8.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
 
    tflite_model = converter.convert()
 
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
 
    size_kb = os.path.getsize(output_path) / 1024
    print(f"INT8 TFLite model saved: {output_path}")
    print(f"Model size: {size_kb:.1f} KB")
    return output_path, size_kb
 
 
def representative_dataset_gen():
    """Yield ~200 samples for INT8 calibration."""
    indices = np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    for i in indices:
        yield [X_train[i:i+1]]
 
 
tflite_path, model_size_kb = convert_to_tflite_int8(
    qat_model,                       # <-- QAT model, not the float baseline
    representative_dataset_gen,
    output_path='ds_cnn_qat_int8.tflite'
)


# INT8 eval

class TFLitePredictor:
    """Wrapper for TFLite INT8 inference."""

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.input_scale  = self.input_details['quantization'][0]
        self.input_zp     = self.input_details['quantization'][1]
        self.output_scale = self.output_details['quantization'][0]
        self.output_zp    = self.output_details['quantization'][1]

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
print("INT8 POST-TRAINING QUANTIZED MODEL")
print("=" * 60)
print(classification_report(y_test, y_pred_int8, target_names=CLASS_NAMES, digits=4))

int8_acc = np.mean(y_pred_int8 == y_test)
print(f"Test Accuracy: {int8_acc:.4f}")
print(f"Accuracy drop vs FP32: {(baseline_acc - int8_acc) * 100:.2f}%")


# Latency enchmarking

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

model.save('ds_cnn_f32.keras')
f32_size_kb = os.path.getsize('ds_cnn_f32.keras') / 1024

print("\n" + "=" * 60)
print("BENCHMARK COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'Float32':>12} {'INT8 PTQ':>12} {'Ratio':>10}")
print("-" * 60)
print(f"{'Accuracy':<25} {baseline_acc:>11.4f} {int8_acc:>11.4f} {int8_acc/baseline_acc:>9.4f}x")
print(f"{'Model Size (KB)':<25} {f32_size_kb:>11.1f} {model_size_kb:>11.1f} {f32_size_kb/model_size_kb:>9.1f}x")
print(f"{'Latency Mean (ms)':<25} {f32_latency['mean_ms']:>11.3f} {int8_latency['mean_ms']:>11.3f} {f32_latency['mean_ms']/int8_latency['mean_ms']:>9.1f}x")
print(f"{'Latency P95 (ms)':<25} {f32_latency['p95_ms']:>11.3f} {int8_latency['p95_ms']:>11.3f}")
print(f"{'Parameters':<25} {model.count_params():>11,}")
