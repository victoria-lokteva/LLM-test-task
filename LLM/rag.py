import warnings

warnings.filterwarnings("ignore")

from templates import data_description, basic_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All

question = "What are the best times of day for ad clicks? List the top 5 hours by their click success rate"



def rag_pipeline(
        question: str,
        template: str,
        data_description_template: str = None,
        csv_file_path: str = None,
        llm_path: str = "/Users/victorialokteva/LLMtesttask/models/mistral.gguf",
        extra_template: str = None
):
    if csv_file_path is None:
        csv_file_path = "/Users/victorialokteva/LLMtesttask/data/dataset.csv"

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

    # добавим шаблон с описанием данных
    if data_description_template:
        template = data_description_template + template
    if extra_template:
        template = template + extra_template
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


    llm = GPT4All(
        model=llm_path,
        backend="llama",
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


def result_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(data)


if __name__ == "__main__":
    result = rag_pipeline(question,
                          basic_template,
                          data_description)


    output_file = f"../generated_code/{question[:20]}.txt"

    result_to_file(result["result"], output_file)
