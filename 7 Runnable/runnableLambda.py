from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
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

joke_chain = RunnableSequence(prompt1, model, parser)

def word_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'words': RunnableLambda(word_counter)
})

chain = RunnableSequence(joke_chain, parallel_chain)

result = chain.invoke({'topic': 'virat kohli'})
# print(result['joke'])
# print(result['words'])

final_result ="""joke - {} \n word count - {}""".format(result['joke'], result['words'])

print(final_result)