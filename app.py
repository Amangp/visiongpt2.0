import gradio as gr
import torch
from inference import VQAInference
from PIL import Image
import os

# Initialize the model
print("🚀 Loading VQA model...")
MODEL_PATH = os.environ.get("VQA_MODEL_PATH", "vqa_transformer.pth")
WORD2IDX_PATH = os.environ.get("VQA_WORD2IDX_PATH", "word2idx.pkl")
ANS2IDX_PATH = os.environ.get("VQA_ANS2IDX_PATH", "ans2idx.pkl")

print(f"📦 Using model checkpoint: {MODEL_PATH}")
vqa_model = VQAInference(
    model_path=MODEL_PATH,
    word2idx_path=WORD2IDX_PATH,
    ans2idx_path=ANS2IDX_PATH,
)

def answer_question(image, question):
    """
    Process image and question, return answer.
    
    Args:
        image: PIL Image from Gradio
        question: str, user's question
        
    Returns:
        str: Formatted answer with confidence scores
    """
    if image is None:
        return "⚠️ Please upload an image first!"
    
    if not question or question.strip() == "":
        return "⚠️ Please enter a question!"
    
    try:
        # Get predictions
        predictions = vqa_model.predict(image, question, top_k=5)
        
        # Format output
        output = f"**Question:** {question}\n\n**Top Predictions:**\n\n"
        for i, (answer, confidence) in enumerate(predictions, 1):
            confidence_pct = confidence * 100
            bar = "█" * int(confidence_pct / 5)  # Visual bar
            output += f"{i}. **{answer}** - {confidence_pct:.2f}%\n"
            output += f"   {bar}\n\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Example questions for users to try
example_questions = [
    "What color is the object?",
    "How many people are in the image?",
    "What is the person doing?",
    "What animal is shown?",
    "What is the weather like?",
    "Is this indoors or outdoors?",
    "What room is this?",
    "What sport is being played?"
]

# Create Gradio interface
with gr.Blocks(title="Visual Question Answering", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🖼️ Visual Question Answering System
        
        Upload an image and ask a question about it! The AI model will analyze the image and provide an answer.
        
        **How to use:**
        1. Upload an image using the image box
        2. Type your question in the text box
        3. Click "Get Answer" to see the predictions
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=400
            )
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=2
            )
            
            # Suggested questions
            gr.Markdown("**💡 Suggested questions:**")
            question_examples = gr.Examples(
                examples=[[q] for q in example_questions],
                inputs=[question_input],
                label=None
            )
            
            submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=1):
            output = gr.Markdown(label="Answer", value="Upload an image and ask a question to get started!")
    
    # Event handlers
    submit_btn.click(
        fn=answer_question,
        inputs=[image_input, question_input],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: (None, "", "Upload an image and ask a question to get started!"),
        inputs=[],
        outputs=[image_input, question_input, output]
    )
    
    # Also submit on Enter key
    question_input.submit(
        fn=answer_question,
        inputs=[image_input, question_input],
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        **ℹ️ Note:** This model was trained on the VQA v2.0 dataset. Best results are obtained with:
        - Clear, well-lit images
        - Specific questions about objects, colors, numbers, actions, and locations
        - Questions similar to the training data
        """
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 Starting Gradio Web Interface...")
    print("="*50 + "\n")
    
    demo.launch(
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
