import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter

web_urls = [
    "https://n.news.naver.com/mnews/article/029/0002927209",
    "https://n.news.naver.com/mnews/article/092/0002358620",
    "https://n.news.naver.com/mnews/article/008/0005136824",
]

loader = WebBaseLoader(web_urls)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separator="\n\n"
)
splitted_docs = text_splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(embedding_function=embedding_model)
vector_store.add_documents(splitted_docs)

# Step 5 setup
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

system_prompt = (
    "다음 검색된 맥락을 사용하여 사용자의 질문에 답하세요. "
    "답을 모르면 모른다고 하고, 추측하지 마세요. "
    "답변은 한국어로 간결하고 정확하게 작성하세요.\n\n"
    "<맥락>{context}</맥락>"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    })
    .assign(answer=RunnableParallel({
        "context": lambda x: format_docs(x["context"]),
        "question": lambda x: x["question"]
    }) | prompt | llm | StrOutputParser())
)

# Step 6 run
query = "AI가 게임 개발에 미치는 영향은 무엇인가요?"
response = rag_chain.invoke({"question": query})

print("\n--- Chain Output Keys ---")
print(response.keys())
print("\n--- Context Docs ---")
for doc in response["context"]:
    print(f"- {doc.page_content[:100]}...")
print("\n--- Answer ---")
print(response["answer"])
