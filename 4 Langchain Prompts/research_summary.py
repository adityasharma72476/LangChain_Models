from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm, max_tokens=20)

st.title("Research Summary Generator")
st.text_input("Enter Research Paper Title", key="paper_title")
if st.button("Generate Summary"):
    paper_title = st.session_state.paper_title
    prompt = f"Summarize the research paper titled '{paper_title}' in a concise manner."
    result = model.invoke(prompt)
    st.subheader("Summary:")
    st.write(result.content)
