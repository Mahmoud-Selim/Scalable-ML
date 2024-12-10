import gradio as gr
from transformers import pipeline
import os


key = os.getenv('hf_key')


# Initialize the pipeline
chatbot = pipeline(task="text-generation", model="Mahmoud-Selim/lora_model", token=key)

def clean_response(raw_response):
    """
    Cleans the raw inference response to extract the assistant's reply.
    """
    try:
        generated_text = raw_response[0]['generated_text']
        for message in generated_text:
            if message['role'] == 'assistant':
                return message['content']
    except (KeyError, IndexError, TypeError):
        return "Error: Unable to process the response."
    return "Error: No response generated."

def respond(message, history, system_message, max_tokens):
    """
    Handles the conversation and calls the pipeline.
    """
    # Prepare the input for the pipeline
    input_message = [
        {"role": "system", "content": system_message}
    ]
    for user, assistant in history:
        if user:
            input_message.append({"role": "user", "content": user})
        if assistant:
            input_message.append({"role": "assistant", "content": assistant})
    input_message.append({"role": "user", "content": message})

    # Generate the response
    raw_response = chatbot(input_message, max_new_tokens=max_tokens)
    cleaned_response = clean_response(raw_response)

    # Update history and return the response incrementally
    history.append((message, cleaned_response))
    return cleaned_response

# Define the Gradio interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a smart chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
    ],
    title="Chatbot Application",
    description="A smart chatbot powered by a custom inference model using the Transformers library.",
    theme="compact",
)

if __name__ == "__main__":
    demo.launch()
