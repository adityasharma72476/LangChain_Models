from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI()

parser1 = StrOutputParser()

class structure_feedback(BaseModel):
    sentiment:Literal['Positive','Negative'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=structure_feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment from the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this Negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'Positive', prompt2 | model | parser1 ),
    (lambda x:x['sentiment'] == 'Negative', prompt3 | model | parser1 ),
    RunnableLambda(lambda x: "Could not find Sentiment")
)

feedback1 = "This is a terrible smartphone"

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': feedback1})

print(result)
