import gradio as gr
import os
import torch
import logging
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# Setup logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'gradio_log.log')

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Create user ID
user_id = str(uuid.uuid4())
logger.info(f"New session started with user ID: {user_id}")

base_path = './RAG_models'

tokenizer = AutoTokenizer.from_pretrained(base_path + '/Sft_model', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path + '/Sft_model', trust_remote_code=True).cuda()

def chat(message, history):
    logger.info(f"User ({user_id}) query: {message}")
    for response, history in model.stream_chat(tokenizer, message, history, max_length=2048, top_p=0.7, temperature=1):
        logger.info(f"Response to {user_id}: {response}")
        yield response

# Feedback function
def feedback(satisfaction):
    logger.info(f"User ({user_id}) satisfaction: {satisfaction}")
    return f"Feedback received: {satisfaction}"

with gr.Blocks() as interface:
    chat_interface = gr.ChatInterface(
        chat,
        title="InternLM2-7B_comac_sft",
        description="InternLM is mainly developed by Shanghai AI Laboratory."
    ).queue(1)
    
    feedback_text = gr.Textbox(label="Feedback", interactive=False)

    with gr.Row():
        gr.Button(value="满意", variant="primary").click(fn=lambda: feedback("满意"), inputs=[], outputs=feedback_text)
        gr.Button(value="不满意", variant="secondary").click(fn=lambda: feedback("不满意"), inputs=[], outputs=feedback_text)

interface.launch(share=True, server_port=10066)
