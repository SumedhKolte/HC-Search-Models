import gradio as gr
from sentence_transformers import SentenceTransformer

# Load the model (adjust path if needed)
model = SentenceTransformer("/app/local_model")

def embed_text(text):
    embedding = model.encode([text])[0]
    return str(embedding)

iface = gr.Interface(
    fn=embed_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text to embed..."),
    outputs="text",
    title="Medical Search Embedding Model",
    description="Enter text to get its embedding using the deployed SentenceTransformer model."
)

if __name__ == "__main__":
    iface.launch()