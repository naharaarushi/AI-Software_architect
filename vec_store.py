from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

loader=DirectoryLoader(
    'data',
    glob='**/*.md',
    loader_cls=TextLoader
)


docs=loader.load()

print('DIRECTORY LOADED')

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250
)

chunks=splitter.split_documents(docs)

print("CHUNK 1 : \n")
print(chunks[0].page_content)
print('\n\n')

print("CHUNKS CREATED -> total chunks : ",len(chunks))

embedding_model=MistralAIEmbeddings(model='mistral-embed')

print("CREATING CHROMA DB ")

vec_store=Chroma.from_documents(
    persist_directory='chroma-db',
    embedding=embedding_model,
    documents=chunks
)

print("CHROMA DB CREATED")