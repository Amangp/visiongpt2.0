import os
import gradio as gr
from inference import VQAInference

# ---------------------------------------------------------
# Initialize Model
# ---------------------------------------------------------

print("🚀 Loading TensorFlow VQA model...")

MODEL_PATH = os.environ.get(
    "VQA_MODEL_PATH",
    "vqa_transformer.keras",
)

WORD2IDX_PATH = os.environ.get(
    "VQA_WORD2IDX_PATH",
    "word2idx.pkl",
)

ANS2IDX_PATH = os.environ.get(
    "VQA_ANS2IDX_PATH",
    "ans2idx.pkl",
)

print(f"📦 Using model checkpoint: {MODEL_PATH}")

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
        return "⚠️ Please upload an image first!"

    if question is None or question.strip() == "":
        return "⚠️ Please enter a question!"

    try:

        predictions = vqa_model.predict(
            image,
            question,
            top_k=5,
        )

        output = (
            f"**Question:** {question}\n\n"
            "**Top Predictions:**\n\n"
        )

        for i, (answer, confidence) in enumerate(
            predictions,
            start=1,
        ):

            confidence *= 100

            bar = "█" * int(confidence / 5)

            output += (
                f"{i}. **{answer}** - "
                f"{confidence:.2f}%\n"
            )

            output += f"{bar}\n\n"

        return output

    except Exception as e:

        return f"❌ Error: {str(e)}"


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
    title="Visual Question Answering",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
# 🖼️ Visual Question Answering System

Upload an image and ask a question.

The TensorFlow VQA model will analyze the image and answer your question.
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
                lines=2,
                placeholder="Ask anything...",
            )

            gr.Markdown(
                "**💡 Example Questions**"
            )

            gr.Examples(
                examples=[[q] for q in example_questions],
                inputs=question_input,
            )

            submit = gr.Button(
                "🔍 Get Answer",
                variant="primary",
            )

            clear = gr.Button(
                "🗑️ Clear",
            )

        with gr.Column(scale=1):

            output = gr.Markdown(
                value="Upload an image and ask a question."
            )

    submit.click(
        answer_question,
        [image_input, question_input],
        output,
    )

    question_input.submit(
        answer_question,
        [image_input, question_input],
        output,
    )

    clear.click(
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

Model trained on **VQA v2.0**

Recommended:

- Clear images
- Natural-language questions
- Questions about objects, colors, numbers, actions and scenes
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