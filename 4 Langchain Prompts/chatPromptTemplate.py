from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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


chat_template = ChatPromptTemplate(
    [
        ('system',"You are a helpful {domain} expert"),
        ('human', "Explain {topic} in simple words")
    ]
)

prompt = chat_template.invoke({'domain': 'Cricket Expert', 'topic':'dusra'})

# print(prompt)

result = model.invoke(prompt)

print(result.content)