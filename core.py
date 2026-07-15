from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embed=MistralAIEmbeddings(model='mistral-embed')
llm=ChatMistralAI(model='mistral-small-latest')

print("LLM AND EMBED MODEL READY")

vec_store=Chroma(
    persist_directory='chroma-db',
    embedding_function=embed
)

print("VEC STORE READY")

retriever=vec_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k' : 8,
        'lambda_mul':0.3,
        'fetch_k' :20
    }
)

print("RETRIEVER READY")

with open ('system.txt') as f:
    system=f.read()


prompt=ChatPromptTemplate.from_messages([
    ('system',system),
    ('human','''Context : {context} 
     
     user requirments
     {question}''')
])
classifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query classifier.

Return ONLY ONE WORD.

ARCHITECTURE:
- System design requests
- App ideas
- Software development
- APIs
- Databases
- Cloud
- Scalability
- Technical architecture

GENERAL:
- Greetings
- Jokes
- Personal questions
- General knowledge
- Small talk

Examples:

Hi -> GENERAL
Hello -> GENERAL
How are you -> GENERAL
Tell me a joke -> GENERAL

Design Netflix -> ARCHITECTURE
Build a food delivery app -> ARCHITECTURE
Create a social media platform -> ARCHITECTURE
Design a payment gateway -> ARCHITECTURE
"""
    ),
    ("human", "{query}")
])

classifier_chain = classifier_prompt | llm


print('RAG IS ACTIVATED PRESS 0 TO EXIT')

while True:

    query = input("YOU : ")

    if query == "0":
        break

    classification = classifier_chain.invoke(
        {"query": query}
    ).content.strip()
    
    chunks_retrieved=retriever.invoke(query)
    print("\nRetrieved Sources:")

    if "ARCHITECTURE" not in classification.upper():
        print("I can only help with software architecture and system design.")
        continue

    for doc in chunks_retrieved:
        print(doc.metadata['source'])


    context= context = "\n\n".join(
    [
        f"Source: {doc.metadata['source']}\n{doc.page_content}"
        for doc in chunks_retrieved
    ]
)
    final_prompt=prompt.invoke({
        'context':context,
        'question':query
    })
    response=llm.invoke(final_prompt)

    print(f'\n AI : {response.content}')









