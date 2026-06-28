from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
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
    template="Generate a joke on the {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="explain the following joke \n {joke}",
    input_variables=['joke']
)


joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explain_joke': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_chain, parallel_chain)

result = chain.invoke({'topic': 'AI'})
print(result['joke'])
print(result['explain_joke'])