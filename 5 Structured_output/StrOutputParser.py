from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm, max_tokens=20)


#1st prompt -> detailed report

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
) 
#2nd prompt -> 5 line summary

template2 = PromptTemplate(
    template='Write a 5 line summary on the folllowing text \n {text}',
    input_variables=['text']
) 

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':"Black hole"})

print(result)