import warnings
from utils import result_to_file
import mytoken
from examples import example11, example8
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.config import Config
from templates import data_description, basic_template, chain_of_thought, code_recommendation, task_description

warnings.filterwarnings("ignore")


class Agent:

    def __init__(self, model: str = "gpt-4", temperature: int = 0):
        self.model = model
        self.temperature = temperature

    def get_response(self, question: str, prompt: str, csv_file_path: str = None):
        if csv_file_path is None:
            csv_file_path = Config().data_paths['dataset']

        db = self.csv_to_embeddings(csv_file_path)

        llm = ChatOpenAI(model=self.model, temperature=self.temperature)

        retriever = db.as_retriever()
        # retriever.search_kwargs = {'k': 10}

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
                    examples: list = None,
                    is_need_coding: bool = True) -> str:
        if is_need_coding:
            template = code_recommendation + template

        # add template with data description
        if data_description_template:
            template = data_description_template + template
        # prompt-engineering with other templates
        if examples:
            for example in examples:
                template = template + "Here is an example of solving similar task " + example

        if extra_template:
            template = template + extra_template
        if use_chain_of_thought:
            template = template + chain_of_thought

        prompt_template = PromptTemplate.from_template(template)
        return prompt_template

    @staticmethod
    def csv_to_embeddings(csv_file_path: str, db: str = 'chroma'):

        # read csv file using langchain
        loader = CSVLoader(file_path=csv_file_path)
        data = loader.load()

        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        docs = text_splitter.split_documents(data)

        if db == 'chroma':
            embedding = OpenAIEmbeddings()
            db = Chroma.from_documents(documents=docs, embedding=embedding)
        elif db == 'faiss':
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


if __name__ == "__main__":
    agent = Agent()
    question_example = """Which types of devices get the most ad clicks, ranked from highest to lowest"""

    prompt = Agent.edit_prompt(question_example, task_description,
                               data_description, use_chain_of_thought=True,
                               is_need_coding=True)

    result = Agent().get_response(question_example, prompt)
    output_file = f"../generated_answers_LLM/{question_example[:20]}.txt"
    result_to_file(result, output_file)

    question_example2 = """Build a model to predict ad click probabilities"""

    prompt = Agent.edit_prompt(question_example2, task_description,
                               data_description, use_chain_of_thought=True, examples=[example11, example8],
                               is_need_coding=True)

    result = Agent().get_response(question_example2, prompt)
    output_file = f"../generated_answers_LLM/{question_example2[:20]}.txt"
    result_to_file(result, output_file)

    question_example = """Identify Important Factors: Find out what factors are most likely to make someone click on online ads.
    Tell me which factors are the strongest predictors and rank them."""

    prompt = Agent.edit_prompt(question_example, task_description,
                               data_description, use_chain_of_thought=True,
                               is_need_coding=True)

    result = Agent().get_response(question_example, prompt)
    output_file = f"../generated_answers_LLM/{question_example[:20]}.txt"
    result_to_file(result, output_file)
