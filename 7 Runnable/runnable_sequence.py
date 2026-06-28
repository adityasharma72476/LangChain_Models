from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm, max_tokens=20)

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain the following joke \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'cricket'})

print(result)
