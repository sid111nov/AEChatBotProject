import os
import requests
from mistralai import Mistral
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import  AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain_mistralai import ChatMistralAI
# from transformers.pipelines import pipeline

import streamlit as st

load_dotenv(os.path.join('env', 'tokens.config'))
token = os.getenv('mistral')
model = 'open-mistral-nemo'

# headers = {"Authorization": f"Bearer {token}"}
# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-hf"


client = Mistral(api_key=token)

st.title("LLM Question Answering App")



# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B",token=token)


# # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small",token=token)

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",token=token)


# text_gen_pipeline = pipeline("text-generation",
#                     model=model,
#                     tokenizer=tokenizer,
#                     max_length=200,
#                     temperature=0.7,
#                     top_p=0.9,
#                     truncation=True
#                     )


# llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

question = st.text_input("Enter your question")

# def query_huggingface_api(question):
#     data = {
#         "inputs":question
#     }
#     response = requests.post(API_URL, headers=headers, json=data)
#     return response.json()

def chat_mistral(question):
    chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]
    )
    return chat_response.choices[0].message.content
           

if question:
    # response = llm.invoke(question)
    response = chat_mistral(question)
    st.write(response)
        



