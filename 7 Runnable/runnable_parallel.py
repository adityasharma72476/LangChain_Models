from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm, max_tokens=20)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet for X plateform about {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Generate a tweet for LinkedIn about {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, model, parser),
    'post':RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': "AI"})

print(result['tweet'])
print(result['post'])
