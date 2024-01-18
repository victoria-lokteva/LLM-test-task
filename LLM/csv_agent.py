from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from templates import coding_template
import mytoken
from examples import example11
import re
import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
from config.config import Config
from utils import result_to_file


class SimpleCSVAgent():

    def __init__(self, csv_file_path: str, temperature: int = 0):
        self.agent = create_csv_agent(ChatOpenAI(model='gpt-4',
                                                 temperature=temperature),
                                      csv_file_path,
                                      verbose=True,
                                      agent_type=AgentType.OPENAI_FUNCTIONS,
                                      max_iterations=10
                                      )

    def get_response(self, prompt):
        return self.agent.invoke(prompt)['output']



if __name__ == "__main__":
    csv_file_path = Config().data_paths['dataset']
    prompt = """Which day of the week sees the highest click rates on our ads?"""


    agent = SimpleCSVAgent(csv_file_path)
    answer = agent.get_response(prompt)

    result_to_file(answer, f"../SimpleAgentCode/{prompt[:20]}.txt")


