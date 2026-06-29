from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

loader = PyPDFLoader('main_preview.pdf')

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
    template="summarize the following document{docs}",
    input_variables=['docs']
)

chain = prompt | model | parser

result = chain.invoke({'docs':docs[0]})

print(result)