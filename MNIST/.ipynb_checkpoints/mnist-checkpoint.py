import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_datasets import load

def normalize_image(image, label):
    return tf.cast(image, tf.float32)/255, label

# Loading data
(train_ds, test_ds), ds_info = load('mnist', as_supervised=True, split=['train', 'test'], with_info=True)

# Preparing training data
train_ds = train_ds.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.cache()
trainb_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
train_ds = train_ds.batch(128)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

# Preparing testing data
test_ds = test_ds.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.batch(128)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    optimizer=tf.keras.optimizers.Adam()
)
history = model.fit(
    train_ds,
    epochs=6,
    validation_data=test_ds
)

# Plotting the results
plt.figure()
plt.plot(history['loss'], label='Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()