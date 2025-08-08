import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize conversation with instruction prompt once
conversation_context = "You are an AI designed to argue and debate. Always challenge the user's statements.\n"

def get_ai_response(user_input):
    global conversation_context

    # Append new user input to conversation
    conversation_context += f"User: {user_input}\nAI:"

    encoded_input = tokenizer(conversation_context, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # Generate response
    chat_history_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode only new tokens generated (after input length)
    response = tokenizer.decode(chat_history_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # Append AI response to conversation for next turn
    conversation_context += f" {response}\n"

    return response.strip()

def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return

    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    def fetch_and_display():
        ai_response = get_ai_response(user_input)
        chat_window.insert(tk.END, f"AI: {ai_response}\n\n")
        chat_window.see(tk.END)

    threading.Thread(target=fetch_and_display).start()

root = tk.Tk()
root.title("Argument AI Partner")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
chat_window.pack(padx=10, pady=10)

entry = tk.Entry(root, width=50)
entry.pack(side=tk.LEFT, padx=10, pady=10)
entry.bind("<Return>", send_message)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
