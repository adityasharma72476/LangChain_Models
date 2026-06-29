from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Research Papers',
    glob='*.pdf',
    loader_cls=PyPDFLoader   
)

docs = loader.load()



print(docs[0].page_content)
print(docs[0].metadata)
print(docs[7].page_content)


#with lazyLoad() 
# docs = loader.lazy_load()

# for document in docs:
#     print(document.metadata)