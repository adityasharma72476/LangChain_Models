from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm, max_tokens=20)

class person(BaseModel):

    name:str = Field(description= "Name of the person")
    age:int = Field(description= "Age of the person", ge=18)
    city: str = Field(description="Name of the city person belongs")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template="generate the name, age and city of the fictional {place} person\n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'Indian'})

print(final_result)
