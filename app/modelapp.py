import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import torch.nn.functional as F

model = DistilBertForSequenceClassification.from_pretrained('./model/fine-tuned-model')
tokenizer = DistilBertTokenizer.from_pretrained('./model/tokenizer')

categories = {
    'INVENTORY, SUPPLIES AND EQUIPMENT': 0,
    'PROFESSIONAL SERVICES': 1,
    'TRANSPORTATION AND TRAVEL': 2,
    'UTILITIES': 3,
    'EMPLOYEE BENEFITS AND COMPENSATION': 4,
    'MEALS AND ENTERTAINMENT': 5,
    'TAX PAYMENTS': 6,
    'LEGAL AND COMPLIANCE FEES': 7,
    'BUSINESS DEVELOPMENT AND INVESTMENT': 8
}

st.title("Text Classification Model")

st.header("Model Description")
st.write("This model is a fine-tuned version of the distilbert-base-uncased model on Hugging Face. DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts using the BERT base model.")
st.write("The model is trained to classify payment reasons into one of the following categories:")
st.write(categories)

st.markdown("[Read more about DistilBert base model here](https://huggingface.co/distilbert/distilbert-base-uncased)")


st.header("Model Parameters and configuration")
st.write(model)


st.header("Try it out")

def predict(model, tokenizer, text):
  inputs = tokenizer(text, truncation=True, padding='max_length', max_length=20, return_tensors='pt')

  outputs = model(**inputs)
  logits = outputs.logits

  prbs = F.softmax(logits, dim=-1)
  predicted_label = torch.argmax(prbs, dim=-1).item()
  for key, value in categories.items():
      if value == predicted_label:
          st.write("The predicted label is:", key)

  return prbs, predicted_label

text = st.text_input("Enter sequence to classify")
if st.button("Classify"):
    predict(model=model, tokenizer=tokenizer, text=text)