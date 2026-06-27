from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-pro',
                   temperature=0.7,
                   max_tokens=10)

result = model.invoke('Who is Virat Kohli?')

print(result.content)