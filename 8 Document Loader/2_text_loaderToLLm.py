from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate summary for the following poem {poem}.",
    input_variables=['poem']
)

chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})

print(result)