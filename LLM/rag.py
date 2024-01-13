from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFacePipeline
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import GPT4All

question = "What are the best times of day for ad clicks? List the top 5 hours by their click success rate"
template = """Use the following pieces of context to answer the question at the end.
    {context}
    Question: {question}
    Helpful Answer:"""


def rag_pipeline(question: str,
                 template: str,
                 csv_file_path: str = "/Users/victorialokteva/LLM-test-task/data/dataset.csv",
                 ):
    # прочтем csv файл с помощью langchain
    loader = CSVLoader(file_path=csv_file_path)
    data = loader.load()
    # разобъем документ на части
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)

    # части докумена превратим в эмбеддинги
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # используем векторное хранилище faiss
    db = FAISS.from_documents(docs, embeddings)

    searchDocs = db.similarity_search(question)

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    model = GPT4All(model="./models/mistral-7b-openorca.Q4_0.gguf", n_threads=8)

    # tokenizer = AutoTokenizer.from_pretrained("model/google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("./models/mistral-7b-openorca.Q4_0.gguf")
    # model = AutoModelForSeq2SeqLM.from_pretrained("model/google/flan-t5-large")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 0, "max_length": 512},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    print(result["result"])
    return result

    ###############


if __name__ == 'main':
    rag_pipeline(question,
                 template)
