"""
Pre-download Vision Transformer (ViT-B/16) weights for TensorFlow.
"""

import gc
import tensorflow as tf
import tensorflow_hub as hub

print("Downloading ViT-B/16 weights...")
print("This may take a few minutes...")

try:
    # Clear any existing TensorFlow session
    tf.keras.backend.clear_session()
    gc.collect()

    # Download and load pretrained ViT model
    vit = hub.KerasLayer(
        "https://tfhub.dev/sayakpaul/vit_b16_fe/1",
        trainable=False,
    )

    # Build the layer with a dummy input
    dummy = tf.random.uniform((1, 224, 224, 3))
    features = vit(dummy)

    print("✅ ViT weights downloaded successfully!")
    print(f"Output feature shape: {features.shape}")

    # Cleanup
    del vit
    tf.keras.backend.clear_session()
    gc.collect()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()