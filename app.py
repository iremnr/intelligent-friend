from flask import Flask, render_template, request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from transformers import pipeline

app = Flask(__name__)

# model_name = "google/pegasus-xsum"
# tokenizer = PegasusTokenizer.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

#summarizer = pipeline("summarization", model="Falconsai/text_summarization")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        #tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=1024).to(device)
        #summary_ = model.generate(tokenized_text, min_length=30, max_length=1024)
        summary =summarizer(input_text, max_length=2000, min_length=180, do_sample=False)
        #summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

        # '''
        #     text = <start> i am yash <end>
        #     vocab = { i: 1, am : 2, yash: 3, start 4}

        #     token = [i, am ,yash]
        #     encode = [1 2, 3, 4]

        #     summary_ = [[4, 3,1, 5]]

        #     summary = yash i

        
        # '''

    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()

