from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

chatmodel =ChatOpenAI(model = 'gpt-5.4-nano')

result = chatmodel.invoke('What is the capital of India?')

print(result.content)