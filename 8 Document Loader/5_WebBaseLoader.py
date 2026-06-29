from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
url = 'https://reference.langchain.com/python/langchain-community/document_loaders/pdf/PyPDFLoader'
loader = WebBaseLoader(url)

doc = loader.load()

print(doc[0].page_content)