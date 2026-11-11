import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="My AI App")
st.title("SENTIMENT AND WORD GUESSING APP")



#Load each model only once, then keep it in memory (this saves time and memory)
@st.cache_resource
def load_seniment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource
def load_fill_mask_model():
    return pipeline("fill-mask", model= "distilbert-base-uncased")


#The special symbol that hides a word (it is [MASK] for this model
MASK = "[MASK]"