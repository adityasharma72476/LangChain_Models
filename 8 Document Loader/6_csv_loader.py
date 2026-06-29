from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='creditcard.csv')

docs = loader.load()

print(docs[1].page_content)
print(docs[1].metadata)
# print(len(docs))