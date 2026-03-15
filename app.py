import wikipediaapi
from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyChatBot/1.0'
)

tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


chat_history_ids = None


def generate_response(user_input, max_new_tokens=100, temperature=0.8, top_p=0.9):
    global chat_history_ids

    new_inputs = tokenizer(
        user_input + tokenizer.eos_token,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    input_ids = new_inputs['input_ids'].to(device)
    attention_mask = new_inputs['attention_mask'].to(device)


    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)

        attention_mask = torch.cat(
            [torch.ones(chat_history_ids.shape, dtype=torch.long).to(device), attention_mask],
            dim=-1
        )

    chat_history_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=top_p,
        temperature=temperature
    )

    reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply


def get_wikipedia_text(query):

    page = wiki.page(query)

    if not page.exists():
        return ""

    text = page.text[:500]

    return text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        user_question = request.form.get("question", "").strip()
        if not user_question:

            return redirect(url_for("index"))

        wiki_context = get_wikipedia_text(user_question)

        prompt = f"""
        Use the following Wikipedia information to answer the question.

        Wikipedia:
        {wiki_context}

        Question:
        {user_question}

        Answer:
        """

        answer = generate_response(prompt)

        return render_template("index.html", answer=answer, question=user_question)

    return render_template("index.html")


if __name__ == "__main__":

    app.run(debug=True)