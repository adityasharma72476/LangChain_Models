from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
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
    template="Generate a detailed report about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following report \n {report}",
    input_variables=['topic']
)

joke_chain = RunnableSequence(prompt1, model, parser)

conditional_chain = RunnableBranch(
    (lambda x: len(x.split()) >500 , RunnableSequence(prompt2, model, parser)),
    (lambda x: len(x.split()) <500 , RunnablePassthrough()),
    RunnableLambda(lambda x: "could not get the report")
)

chain = RunnableSequence(joke_chain, conditional_chain)

result = chain.invoke({'topic': 'Use of AI in modern era'})
print(result)