from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

load_dotenv()



model = ChatOpenAI()

#Data Format--- Schema
class review(TypedDict):
    summary:Annotated[str, "A brief summary"]
    sentiment:str

structured_model = model.with_structured_output(review)

reviews = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI
looks outdated compared to other brands. Hoping for a software update to fix this."""

result = structured_model.invoke(reviews)

print(result)