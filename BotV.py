import os
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# API keys
OPENAI_API_KEY = 'OPENAI_API_KEY'
PINECONE_API_KEY = 'PINECONE_API_KEY'

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud='aws', region='us-west-2')

index_name = 'langchain-retrieval-augmentation-fast'

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=1536,  # Dimensionality of text-embedding-ada-002
    metric='dotproduct',
    spec=spec
)

index = pc.Index(index_name)

index_stats = index.describe_index_stats()
print(index_stats)

for batch in dataset.iter_documents(batch_size=100):
    index.upsert(batch)

openai_embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)

vectorstore = Pinecone(index, openai_embed.embed_query, "text")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "YOUR_QUERY"
qa.run(query)
