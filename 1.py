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


root = tk.Tk()
root.title("Argument AI Partner")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
chat_window.pack(padx=10, pady=10)

entry = tk.Entry(root, width=50)
entry.pack(side=tk.LEFT, padx=10, pady=10)

send_button = tk.Button(root, text="Send")
send_button.pack(side=tk.LEFT, padx=5)

root.mainloop()


def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

send_button.config(command=send_message)
entry.bind("<Return>", send_message)


def get_ai_response(user_input):
    # Placeholder echo function
    return "AI: " + user_input[::-1]  # just reverse input as dummy response

def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return

    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    ai_response = get_ai_response(user_input)
    chat_window.insert(tk.END, f"{ai_response}\n\n")
    chat_window.see(tk.END)


def get_ai_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return

    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    ai_response = get_ai_response(user_input)
    chat_window.insert(tk.END, f"AI: {ai_response}\n\n")
    chat_window.see(tk.END)

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


conversation_context = ""

def get_ai_response(user_input):
    global conversation_context

    conversation_context += user_input + tokenizer.eos_token
    inputs = tokenizer.encode(conversation_context, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    conversation_context += response + tokenizer.eos_token
    return response



conversation_context = "You are an AI designed to argue and debate. Always challenge the user's statements.\n"

def get_ai_response(user_input):
    global conversation_context

    conversation_context += f"User: {user_input}\nAI:"
    inputs = tokenizer.encode(conversation_context, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    conversation_context += f" {response}\n"
    return response.strip()


def get_ai_response(user_input):
    global conversation_context

    conversation_context += f"User: {user_input}\nAI:"
    encoded_input = tokenizer(conversation_context, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    conversation_context += f" {response}\n"
    return response.strip()


def reset_chat():
    global conversation_context
    conversation_context = "You are an AI designed to argue and debate. Always challenge the user's statements.\n"
    chat_window.delete(1.0, tk.END)

reset_button = tk.Button(root, text="Reset Chat", command=reset_chat)
reset_button.pack(side=tk.LEFT, padx=5)


def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return

    entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)

    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    def fetch_and_display():
        ai_response = get_ai_response(user_input)
        chat_window.insert(tk.END, f"AI: {ai_response}\n\n")
        chat_window.see(tk.END)
        entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)

    threading.Thread(target=fetch_and_display).start()


def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return

    entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)

    chat_window.insert(tk.END, f"You: {user_input}\n")
    chat_window.insert(tk.END, "AI: (typing...)\n")
    chat_window.see(tk.END)
    entry.delete(0, tk.END)

    def fetch_and_display():
        # Remove typing line
        chat_window.delete("end-2l", "end-1l")
        ai_response = get_ai_response(user_input)
        chat_window.insert(tk.END, f"AI: {ai_response}\n\n")
        chat_window.see(tk.END)
        entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)

    threading.Thread(target=fetch_and_display).start()
