import warnings

from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.config import Config
from templates import data_description, basic_template, chain_of_thought, code_recommendation

warnings.filterwarnings("ignore")


class Agent:

    def __init__(self):
        self.llm_path = Config().llm_paths['mistral']

    def get_response(self, question, prompt, csv_file_path: str = None):
        if csv_file_path is None:
            csv_file_path = Config().data_paths['dataset']

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

    @staticmethod
    def edit_prompt(template: str,
                    data_description_template: str = None,
                    extra_template: str = None,
                    use_chain_of_thought: bool = False,
                    examples: str = None,
                    is_need_coding: bool = True):
        if is_need_coding:
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

    @staticmethod
    def csv_to_embeddings(csv_file_path: str):

        # read csv file using langchain
        loader = CSVLoader(file_path=csv_file_path)
        data = loader.load()

        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)

        # parts of the documents -> embeddings
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
    question_example = """Which types of devices get the most ad clicks, ranked from highest to lowest"""

    agent = Agent()

    prompt = Agent.edit_prompt(question_example, basic_template,
                               data_description, use_chain_of_thought=True)

    result = Agent().get_response(question_example, prompt)
    output_file = f"../generated_code/{question_example[:20]}.txt"
    result_to_file(result, output_file)
