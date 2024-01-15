import warnings
warnings.filterwarnings("ignore")

from templates import data_description, basic_template, chain_of_thought, code_recommendation
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class Agent:

    def __init__(self, llm_path="/Users/victorialokteva/LLMtesttask/models/mistral.gguf"):
        self.llm_path = llm_path

    def get_response(self, prompt, csv_file_path: str = None):
        if csv_file_path is None:
            csv_file_path = "/Users/victorialokteva/LLMtesttask/data/dataset.csv"

        db = self.csv_to_embeddings(csv_file_path)

        llm = GPT4All(
            model=self.llm_path,
            backend="llama",
        )

        retriever = db.as_retriever()
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        result = rag_chain.invoke(question)
        return result

    def edit_prompt(question: str,
                    template: str,
                    data_description_template: str = None,
                    extra_template: str = None,
                    use_chain_of_thought: bool = False,
                    examples: str = None,
                    need_cooding: bool = True
                    ):
        if need_cooding:
            template = code_recommendation + template

        # add template with data description
        if data_description_template:
            template = data_description_template + template
        # prompt-engineering with other templates
        if examples:
            template = template + examples

        if extra_template:
            template = template + extra_template
        if use_chain_of_thought:
            template = template + chain_of_thought

        prompt_template = PromptTemplate.from_template(template)
        return prompt_template

    def csv_to_embeddings(self, csv_file_path):

        # read csv file using langchain
        loader = CSVLoader(file_path=csv_file_path)
        data = loader.load()
        # split documents into chuncks
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
        # use vector store faiss
        db = FAISS.from_documents(docs, embeddings)
        return db


def result_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(data)


if __name__ == "__main__":
    question = """Which types of devices get the most ad clicks, ranked from highest to lowest"""

    agent = Agent()

    prompt = Agent.edit_prompt(question, basic_template,
                               data_description, use_chain_of_thought=True)

    result = Agent().get_response(prompt)
    output_file = f"../generated_code/{question[:20]}.txt"
    result_to_file(result, output_file)
