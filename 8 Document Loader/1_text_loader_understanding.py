from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()
print(docs) #-- it returns a list of document
print(type(docs)) #-- it tells it is a list type

print(len(docs)) #-- length of docs is 1

print(docs[0]) #-- return the document object having 1) 'page content' 2)'metadata'

print(docs[0].page_content) #-- it contains the real poem full text
print(docs[0].page_content) #-- it contains the metadata of poem-- ' metadata={'source': 'cricket.txt'}
print(len(docs[0].page_content.split())) #-- number of words in the poem -- 222

