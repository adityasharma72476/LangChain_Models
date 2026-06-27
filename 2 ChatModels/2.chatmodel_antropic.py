from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model =ChatAnthropic(model = 'claude-3-7-sonnet-latest',
              temperature=0.7)

result = model.invoke('Who is Virat Kohli?')

print(result.content)