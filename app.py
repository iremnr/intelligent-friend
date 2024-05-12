from flask import Flask, render_template, request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline


app = Flask(__name__)

# model_name = "google/pegasus-xsum"
# tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


modified_parameters = {
    "vocab_size": 50265,
    "max_position_embeddings": 1024,
    "encoder_layers": 12,
    "encoder_ffn_dim": 4096,
    "encoder_attention_heads": 16,
    "decoder_layers": 12,
    "decoder_ffn_dim": 4096,
    "decoder_attention_heads": 16,
    "encoder_layerdrop": 0.0,
    "decoder_layerdrop": 0.0,
    "activation_function": "gelu",
    "d_model": 1024,
    "dropout": 0.1,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "init_std": 0.02,
    "classifier_dropout": 0.0,
    "scale_embedding": False,
    "use_cache": True,
    "num_labels": 3,
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "is_encoder_decoder": True,
    "decoder_start_token_id": 2,
    "forced_eos_token_id": 2
}
# from transformers import GPT2Config

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
#tokenizer = BartTokenizer.from_pretrained("facebook/bart-large", **modified_parameters)

# gpt2_config = GPT2Config(
#     num_layers=24,                       # Increase the number of transformer layers
#     d_model=1024,                        # Increase the dimensionality of hidden states
#     num_heads=16,                        # Increase the number of attention heads
#     d_ff=4096,                           # Increase the dimensionality of feed-forward network
#     vocab_size=50257,                    # Size of vocabulary
#     max_position_embeddings=1024,        # Maximum length of input sequences
#     dropout=0.2,                         # Increase dropout probability
#     attention_probs_dropout_prob=0.2,    # Increase dropout probability for attention weights
#     activation_function="gelu_new"      # Use a different activation function (e.g.,Âgelu_new)
# )

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2" , config=gpt2_config).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        #tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        # summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        # summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
        summary = summarizer(input_text, max_length=330, min_length=300, do_sample=False, early_stopping=True)


    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()
