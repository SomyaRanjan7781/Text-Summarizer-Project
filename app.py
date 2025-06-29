import re
import nltk
import gradio as gr
from transformers import pipeline
nltk.download('punkt')
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    return text.strip()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def abstractive_summary(text, max_len=130, min_len=30):
    try:
        text = clean_text(text)
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"[Error] {str(e)}"
gr.Interface(
    fn=abstractive_summary,
    inputs=[
        gr.Textbox(lines=15, label="Enter Long Text Here"),
        gr.Slider(50, 300, value=130, step=10, label="Max Summary Length"),
        gr.Slider(10, 100, value=30, step=5, label="Min Summary Length")
    ],
    outputs="text",
    title="📝 Text Summarizer (SUMEASY)",
    description="Paste a long article, biography, or report and get a concise summary using an abstractive model (facebook/bart-large-cnn)."
).launch()