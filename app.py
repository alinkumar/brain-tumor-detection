import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

from utils import (
    load_model,
    preprocess_image,
    generate_explanation
)


CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]


st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


with open("style.css", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


st.markdown("""
<div class="hero">

<h1>🧠 Brain Tumor Detection AI</h1>

<p>
AI-Powered Brain Tumor Classification using EfficientNetB0 & Explainable AI
</p>

<div class="hero-badges">

<span>TensorFlow</span>
<span>Transfer Learning</span>
<span>Computer Vision</span>
<span>Medical AI</span>
<span>Explainable AI</span>

</div>

</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


with st.spinner("🧠 Loading AI Model..."):
    model = get_model()


st.write("")


c1, c2, c3 = st.columns(3)


with c1:
    st.markdown("""
    <div class="card">
        <div class="icon">🧠</div>
        <h3>Architecture</h3>
        <h2>EfficientNetB0</h2>
    </div>
    """, unsafe_allow_html=True)


with c2:
    st.markdown("""
    <div class="card">
        <div class="icon">📊</div>
        <h3>Classification</h3>
        <h2>4 Classes</h2>
    </div>
    """, unsafe_allow_html=True)


with c3:
    st.markdown("""
    <div class="card">
        <div class="icon">⚡</div>
        <h3>Framework</h3>
        <h2>TensorFlow</h2>
    </div>
    """, unsafe_allow_html=True)


st.write("")


st.markdown("""
<div class="upload-card">

<h2>📤 Upload MRI Image</h2>

<p>
Supported Formats : JPG • JPEG • PNG
</p>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file:

    image = Image.open(uploaded_file)

    left, right = st.columns([1.2, 1])


    with left:

        st.markdown("## 🖼 MRI Preview")

        st.image(
            image,
            use_container_width=True
        )


    with right:

        st.markdown("## 📋 Image Details")

        st.metric(
            "Width",
            image.size[0]
        )

        st.metric(
            "Height",
            image.size[1]
        )

        st.metric(
            "Color",
            image.mode
        )

        st.success(
            "MRI Loaded Successfully"
        )


    img_array = preprocess_image(image)


st.write("")


if uploaded_file:

    st.markdown("---")

    _, center, _ = st.columns([1, 2, 1])


    with center:

        analyze = st.button(
            "🧠 Analyze MRI Scan",
            use_container_width=True,
            type="primary"
        )


    if analyze:

        with st.spinner(
            "🧠 AI Model is analyzing the MRI scan..."
        ):

            prediction = model.predict(
                img_array,
                verbose=0
            )


        probabilities = prediction[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index] * 100
        )


        st.markdown("---")

        st.markdown("## 🎯 Prediction Result")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Predicted Class",
                predicted_class
            )


        with col2:

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


        probability_df = pd.DataFrame({

            "Tumor Type": CLASS_NAMES,

            "Probability": np.round(
                probabilities * 100,
                2
            )

        })


        fig = px.bar(

            probability_df,

            x="Probability",

            y="Tumor Type",

            orientation="h",

            color="Probability",

            text="Probability",

            color_continuous_scale="Blues"

        )


        fig.update_layout(

            template="plotly_dark",

            height=420,

            coloraxis_showscale=False,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            xaxis_title="Probability (%)",

            yaxis_title=""

        )


        fig.update_traces(

            texttemplate="%{text:.2f}%",

            textposition="outside"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader(
            "📊 Prediction Probability"
        )


        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True
        )


        st.write("")

        if predicted_class == "No Tumor":

            st.success(
                f"""
**Prediction Completed**

The model classified the uploaded MRI as
**{predicted_class}** with **{confidence:.2f}%**
confidence.
"""
            )

        else:

            st.warning(
                f"""
**Prediction Completed**

The model classified the uploaded MRI as
**{predicted_class}** with **{confidence:.2f}%**
confidence.
"""
            )


        st.info(
            "This AI prediction is intended for educational "
            "and AI-assisted analysis only and is not a medical diagnosis."
        )


        st.markdown("---")

        st.markdown(
            "## 🔍 Explainable AI — Integrated Gradients"
        )


        with st.spinner(
            "🔬 Generating model attribution..."
        ):

            original_image, focused_heatmap, overlay = (
                generate_explanation(
                    model,
                    img_array,
                    predicted_index
                )
            )


        xai_col1, xai_col2, xai_col3 = st.columns(3)


        with xai_col1:

            st.markdown(
                "### 🖼 Original MRI"
            )

            st.image(
                original_image,
                use_container_width=True
            )


        with xai_col2:

            st.markdown(
                "### 🔥 Focused Model Attribution"
            )

            st.image(
                focused_heatmap,
                clamp=True,
                use_container_width=True
            )


        with xai_col3:

            st.markdown(
                "### 🎯 Prediction Explanation"
            )

            st.image(
                overlay,
                use_container_width=True
            )


            st.info(
            """
### About the AI Explanation

The highlighted regions indicate the areas of the MRI that most influenced the model's prediction.

This visualization is generated using **Integrated Gradients**, an Explainable AI (XAI) technique.

It helps explain the model's reasoning, but it **does not represent an exact tumor boundary or clinical segmentation**.
"""
        )


        st.markdown("---")

        st.markdown("## 🏥 AI Analysis Dashboard")


        left_panel, right_panel = st.columns(
            [1.2, 1]
        )


        with left_panel:

            st.markdown(
                "### 📋 Prediction Summary"
            )

            st.metric(
                "Predicted Class",
                predicted_class
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            sorted_probabilities = np.sort(probabilities)

            prediction_margin = (
            sorted_probabilities[-1]
            - sorted_probabilities[-2]
            ) * 100

            st.metric(
            "Prediction Margin",
            f"{prediction_margin:.2f}%"
            )

            st.metric(
                "Model",
                "EfficientNetB0"
            )

            st.metric(
                "Loss Function",
                "Focal Loss"
            )


        with right_panel:

            st.markdown(
                "### 🎯 Confidence Assessment"
            )

            if confidence >= 90:

                st.success(
                    "🟢 Very High Confidence"
                )

                st.caption(
                    "The model strongly favors the predicted class over all remaining classes."
                )

            elif confidence >= 80:

                st.success(
                    "🟢 High Confidence"
                )

                st.caption(
                    "The prediction is reliable, although clinical confirmation is recommended."
                )

            elif confidence >= 65:

                st.warning(
                    "🟡 Moderate Confidence"
                )

                st.caption(
                    "The prediction is reasonably confident, but alternative classes also received noticeable probability."
                )

            elif confidence >= 50:

                st.warning(
                    "🟠 Low Confidence"
                )

                st.caption(
                    "The model shows uncertainty because multiple classes received similar probabilities."
                )

            else:

                st.error(
                    "🔴 Very Low Confidence"
                )

                st.caption(
                    "The model is uncertain. This prediction should be interpreted cautiously."
                )

            st.metric(
                "Input Size",
                "224 × 224"
            )

            st.metric(
                "Classes",
                len(CLASS_NAMES)
            )


        st.write("")

        st.markdown("---")

        st.markdown(
            "## 🩺 AI Classification Interpretation"
        )


        st.markdown(
            f"""
### Predicted Class: {predicted_class}

The EfficientNetB0 model classified the uploaded MRI
as **{predicted_class}** with a model confidence of
**{confidence:.2f}%**.


The Integrated Gradients visualization above provides
an explanation of the image regions that contributed
to this prediction.

**Clinical confirmation by a qualified medical
professional is recommended.**
"""
        )

        st.info(
    """
This AI system is intended for educational purposes and AI-assisted MRI interpretation.

It should not be used as a substitute for professional medical diagnosis.
"""
)


        with st.expander(
            "⚙ Technical Details",
            expanded=False
        ):

            st.write(
                "Architecture : EfficientNetB0"
            )

            st.write(
                "Framework : TensorFlow"
            )

            st.write(
                "Loss Function : Sparse Categorical Focal Loss"
            )

            st.write(
                "Explainability : Integrated Gradients"
            )

            st.write(
                "Image Resolution : 224 × 224"
            )

            st.write(
                f"Predicted Class : {predicted_class}"
            )

            st.write(
                f"Prediction Confidence : {confidence:.2f}%"
            )

            st.markdown(
"""
<div class='footer'>

Made with ❤️ using TensorFlow • Streamlit • Explainable AI

</div>
""",
unsafe_allow_html=True
)