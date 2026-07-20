import os
import gradio as gr
from inference import VQAInference

# ---------------------------------------------------------
# Initialize Model
# ---------------------------------------------------------

print("🚀 Loading TensorFlow VQA model...")

MODEL_PATH = os.environ.get(
    "VQA_MODEL_PATH",
    "vqa_transformer.weights.h5",
)

WORD2IDX_PATH = os.environ.get(
    "VQA_WORD2IDX_PATH",
    "data/processed/word2idx.pkl",
)

ANS2IDX_PATH = os.environ.get(
    "VQA_ANS2IDX_PATH",
    "data/processed/ans2idx.pkl",
)

print(f"📦 Model checkpoint : {MODEL_PATH}")
print(f"📚 Vocabulary       : {WORD2IDX_PATH}")
print(f"🏷️ Answer mapping   : {ANS2IDX_PATH}")

vqa_model = VQAInference(
    model_path=MODEL_PATH,
    word2idx_path=WORD2IDX_PATH,
    ans2idx_path=ANS2IDX_PATH,
)

# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def answer_question(image, question):

    if image is None:
        return "⚠️ Please upload an image."

    if not question or question.strip() == "":
        return "⚠️ Please enter a question."

    try:

        predictions = vqa_model.predict(
            image=image,
            question=question,
            top_k=5,
        )

        output = f"## Question\n{question}\n\n"
        output += "## Top Predictions\n\n"

        for i, (answer, confidence) in enumerate(predictions, start=1):

            confidence *= 100

            output += (
                f"**{i}. {answer}** "
                f"({confidence:.2f}%)\n\n"
            )

        return output

    except Exception as e:
        return f"❌ {e}"

# ---------------------------------------------------------
# Example Questions
# ---------------------------------------------------------

example_questions = [
    "What color is the object?",
    "How many people are in the image?",
    "What is the person doing?",
    "What animal is shown?",
    "What is the weather like?",
    "Is this indoors or outdoors?",
    "What room is this?",
    "What sport is being played?",
]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

with gr.Blocks(
    title="TensorFlow Visual Question Answering",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
# 🖼️ Visual Question Answering

Upload an image and ask a question about it.

The model is trained on the **VQA v2** dataset.
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=400,
            )

            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask anything about the image...",
                lines=2,
            )

            gr.Markdown("### Example Questions")

            gr.Examples(
                examples=[[q] for q in example_questions],
                inputs=question_input,
            )

            submit_btn = gr.Button(
                "🔍 Predict",
                variant="primary",
            )

            clear_btn = gr.Button("🗑️ Clear")

        with gr.Column(scale=1):

            output = gr.Markdown(
                "Upload an image and ask a question."
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[image_input, question_input],
        outputs=output,
    )

    question_input.submit(
        fn=answer_question,
        inputs=[image_input, question_input],
        outputs=output,
    )

    clear_btn.click(
        lambda: (
            None,
            "",
            "Upload an image and ask a question.",
        ),
        outputs=[
            image_input,
            question_input,
            output,
        ],
    )

    gr.Markdown(
        """
---

### Recommended Questions

- What color is the object?
- How many people are there?
- What is the person doing?
- What animal is shown?
- Is it indoors or outdoors?
"""
    )

# ---------------------------------------------------------
# Launch
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("🌐 Starting TensorFlow VQA")
    print("=" * 60)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )