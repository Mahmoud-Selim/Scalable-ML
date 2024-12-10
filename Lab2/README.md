# Fine-Tuning and Serving Llama 3.2-1B Parameter Model

This repository contains a project that fine-tunes a Llama 3.2-1B parameter model using a provided notebook and serves it as a chatbot on Hugging Face using Gradio. The script also includes enhancements to improve performance and user experience by incorporating a starting system message and dynamic token limit configuration.

## Features
- Fine-tune the **Llama 3.2-1B** model using LoRA (Low-Rank Adaptation).
- Serve the fine-tuned model on Hugging Face using a Gradio interface.
- Customizable system messages for different chatbot personalities.
- Dynamic control over token limit to manage inference speed.

---

## Project Structure

- **Notebook:**
  - The notebook file (`Another_copy_of_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb`) contains the code and instructions for fine-tuning the Llama 3.2-1B parameter model using LoRA.

- **Serving Script:**
  - The `app.py` script handles serving the model using Hugging Face’s `transformers` library and Gradio.

- **Dependencies:**
  - Gradio: Provides an easy-to-use UI for the chatbot.
  - Transformers: Handles the fine-tuning and inference processes.

---

## Requirements

### Install Dependencies

To set up the environment, ensure you have Python 3.8 or later installed. Then install the necessary dependencies:

```bash
pip install gradio transformers
```

---

## Fine-Tuning the Model

### Steps

1. **Load the Notebook:**
   - Open `Another_copy_of_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb` in Jupyter Notebook or Google Colab.

2. **Prepare Dataset:**
   - Ensure you have a well-prepared dataset for conversational fine-tuning.
   - Follow the instructions in the notebook to preprocess and load the data.

3. **Fine-Tune the Model:**
   - Use the notebook’s cells to fine-tune the Llama 3.2-1B model using the LoRA technique.
   - Save the fine-tuned model.

4. **Upload to Hugging Face:**
   - Use Hugging Face’s model upload feature to host the fine-tuned model. Make sure to note down the model name and access key.

---

## Running the Chatbot

### Serving Script

Save the following script as `app.py`:

```python
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
```

### Run the Script

Run the script in your terminal:

```bash
python app.py
```

Access the chatbot UI at `http://127.0.0.1:7860`.

---

## Key Features of the Script

1. **Customizable System Messages:**
   - Change the personality of the chatbot by modifying the "System message" textbox.

2. **Dynamic Token Limit:**
   - Adjust the maximum token limit with the slider to balance response quality and computation time.

3. **Error Handling:**
   - Ensures graceful handling of invalid responses from the model pipeline.

---

## Notes
- The project uses Hugging Face's API, so you need an API key set as an environment variable (`hf_key`).
- Make sure the model hosted on Hugging Face matches the name provided in the script (`Mahmoud-Selim/lora_model`).
- Performance may vary based on the hardware and input token limits.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For any issues or inquiries, please contact:
- Name: Mahmoud Selim
- Email: [your-email@example.com]

