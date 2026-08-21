"""Load a Keras-2 .h5 under either Keras 2 (tf-keras) or Keras 3.

KNIME's model-trainer.py runs on TensorFlow 2.15-ish, where `tf.keras` is
Keras 2. Keras 2 writes `groups: 1` into every DepthwiseConv2D config (the
MobileNetV2 backbone has 17 of them). Keras 3 dropped that argument, so it
rejects the file with:

    Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}

Two independent defences, because Streamlit Cloud can silently give us
either Keras:
  1. TF_USE_LEGACY_KERAS=1 routes tf.keras back to Keras 2 (needs the
     `tf-keras` package). Only works if it is set BEFORE tensorflow is
     imported - hence this module must be the first import in the app.
  2. If that fails anyway (tf-keras missing, cached build, TF>=2.20), retry
     the load with a DepthwiseConv2D subclass that drops the stale key.
"""

import os

# Must precede `import tensorflow` anywhere in the process.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def keras_version() -> str:
    """Version string for the Keras actually backing tf.keras."""
    import tensorflow as tf

    version = getattr(tf.keras, "__version__", None)
    if version:
        return version
    try:  # TF_USE_LEGACY_KERAS routes tf.keras to the tf_keras package
        import tf_keras

        return f"{tf_keras.__version__} (legacy)"
    except ImportError:
        return "unknown"


def load_model(path):
    """Load `path`, tolerating a Keras-2 .h5 opened by Keras 3."""
    import tensorflow as tf

    keras = tf.keras
    try:
        return keras.models.load_model(str(path), compile=False)
    except Exception as first_error:  # noqa: BLE001 - retried below
        if "groups" not in str(first_error):
            raise

        class DepthwiseConv2D(keras.layers.DepthwiseConv2D):
            """DepthwiseConv2D that ignores Keras 2's `groups` key."""

            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("groups", None)
                return super().from_config(config)

        return keras.models.load_model(
            str(path),
            compile=False,
            custom_objects={"DepthwiseConv2D": DepthwiseConv2D},
        )
