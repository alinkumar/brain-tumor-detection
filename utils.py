import tensorflow as tf
import numpy as np
import cv2


CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32)

        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred
        )

        pt = tf.exp(-ce)

        focal_loss = (
            self.alpha
            * tf.pow(1.0 - pt, self.gamma)
            * ce
        )

        return tf.reduce_mean(focal_loss)

    def get_config(self):

        config = super().get_config()

        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })

        return config


def load_model():

    model = tf.keras.models.load_model(
        "best_brain_tumor_model.keras",
        custom_objects={
            "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss
        },
        compile=False
    )

    return model


def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


def integrated_gradients(
    model,
    image,
    target_class,
    steps=64,
    batch_size=4
):

    image = tf.cast(
        image,
        tf.float32
    )

    baseline = tf.zeros_like(
        image
    )

    alphas = tf.linspace(
        0.0,
        1.0,
        steps + 1
    )

    interpolated = (
        baseline
        + alphas[:, None, None, None]
        * (image - baseline)
    )

    gradients_list = []

    for start in range(
        0,
        steps + 1,
        batch_size
    ):

        end = min(
            start + batch_size,
            steps + 1
        )

        batch = interpolated[
            start:end
        ]

        with tf.GradientTape() as tape:

            tape.watch(
                batch
            )

            predictions = model(
                batch,
                training=False
            )

            target = predictions[
                :,
                target_class
            ]

        gradients = tape.gradient(
            target,
            batch
        )

        gradients_list.append(
            gradients
        )

        del batch
        del predictions
        del target
        del gradients

    gradients = tf.concat(
        gradients_list,
        axis=0
    )

    avg_gradients = (
        gradients[:-1]
        + gradients[1:]
    ) / 2.0

    avg_gradients = tf.reduce_mean(
        avg_gradients,
        axis=0
    )

    attributions = (
        image - baseline
    ) * avg_gradients

    attributions = tf.reduce_sum(
        attributions,
        axis=-1
    )

    attributions = tf.maximum(
        attributions,
        0.0
    )

    heatmap = attributions.numpy()[0]

    heatmap = cv2.GaussianBlur(
        heatmap.astype(np.float32),
        (0, 0),
        sigmaX=2.5
    )

    high_value = np.percentile(
        heatmap,
        95
    )

    heatmap = np.clip(
        heatmap / (high_value + 1e-8),
        0.0,
        1.0
    )

    return heatmap


def create_brain_mask(
    original_image
):

    image_uint8 = np.uint8(
        np.clip(
            original_image,
            0.0,
            1.0
        ) * 255
    )

    gray = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (9, 9),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(
        mask
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    clean_mask = cv2.erode(
        clean_mask,
        np.ones(
            (11, 11),
            np.uint8
        ),
        iterations=2
    )

    clean_mask = (
        clean_mask.astype(np.float32)
        / 255.0
    )

    return clean_mask


def create_focused_attribution(
    heatmap,
    original_image
):

    brain_mask = create_brain_mask(
        original_image
    )

    heatmap = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap_clean = (
        heatmap * brain_mask
    )

    heatmap_clean = cv2.GaussianBlur(
        heatmap_clean.astype(np.float32),
        (0, 0),
        sigmaX=3
    )

    positive_values = heatmap_clean[
        heatmap_clean > 0
    ]

    if positive_values.size > 0:

        threshold = np.percentile(
            positive_values,
            90
        )

    else:

        threshold = 0.0

    focused_heatmap = np.where(
        heatmap_clean >= threshold,
        heatmap_clean,
        0.0
    )

    max_value = np.max(
        focused_heatmap
    )

    if max_value > 0:

        focused_heatmap = (
            focused_heatmap
            / max_value
        )

    focused_heatmap = cv2.GaussianBlur(
        focused_heatmap.astype(np.float32),
        (0, 0),
        sigmaX=2
    )

    return np.clip(
        focused_heatmap,
        0.0,
        1.0
    )


def create_colored_heatmap(
    heatmap
):

    heatmap_uint8 = np.uint8(
        np.clip(
            heatmap,
            0.0,
            1.0
        ) * 255
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_TURBO
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    colored_heatmap = (
        colored_heatmap.astype(np.float32)
        / 255.0
    )

    return colored_heatmap


def create_overlay(
    original_image,
    heatmap
):

    original_image = np.clip(
        original_image,
        0.0,
        1.0
    )

    heatmap = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = cv2.GaussianBlur(
        heatmap.astype(np.float32),
        (0, 0),
        sigmaX=2
    )

    colored_heatmap = create_colored_heatmap(
        heatmap
    )

    alpha = np.expand_dims(
        np.power(
            heatmap,
            0.80
        ),
        axis=-1
    )

    overlay = (
        original_image
        * (1.0 - 0.70 * alpha)
        + colored_heatmap
        * (0.70 * alpha)
    )

    return np.clip(
        overlay,
        0.0,
        1.0
    )


def generate_explanation(
    model,
    img_array,
    predicted_class
):

    original_image = (
        img_array[0].copy()
        / 255.0
    )

    heatmap = integrated_gradients(
        model,
        img_array,
        predicted_class,
        steps=64
    )

    focused_heatmap = create_focused_attribution(
        heatmap,
        original_image
    )

    colored_heatmap = create_colored_heatmap(
        focused_heatmap
    )

    overlay = create_overlay(
        original_image,
        focused_heatmap
    )

    return (
        original_image,
        colored_heatmap,
        overlay
    )