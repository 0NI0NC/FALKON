import os
from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ShellTool, DuckDuckGoSearchRun, WikipediaQueryRun
from tempfile import TemporaryDirectory
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

print("""
 /$$$$$$$$ /$$$$$$  /$$       /$$   /$$  /$$$$$$  /$$   /$$
| $$_____//$$__  $$| $$      | $$  /$$/ /$$__  $$| $$$ | $$
| $$     | $$  \ $$| $$      | $$ /$$/ | $$  \ $$| $$$$| $$
| $$$$$  | $$$$$$$$| $$      | $$$$$/  | $$  | $$| $$ $$ $$
| $$__/  | $$__  $$| $$      | $$  $$  | $$  | $$| $$  $$$$
| $$     | $$  | $$| $$      | $$\  $$ | $$  | $$| $$\  $$$
| $$     | $$  | $$| $$$$$$$$| $$ \  $$|  $$$$$$/| $$ \  $$
|__/     |__/  |__/|________/|__/  \__/ \______/ |__/  \__/
""")

class FileWriteTool(BaseTool):

    name: str = "File Write Tool"

    description: str = "A tool for writing content to a file based on the provided task description."

    def __init__(self, task=None):
        super().__init__(task=task)

    def _run(self, argument: str) -> str:
        if not hasattr(self, "task"):
            raise ValueError("Task attribute not found. Make sure the task is passed when initializing the tool.")

        filename = self.task.description.replace(" ", "_") + ".py"

        with open(filename, "w") as file:
            file.write(argument)

        return f"Content has been written to {filename}"

import os

def create_data_folder():
    """
    Checks if the DATA folder exists, if not, creates it.
    """
    if not os.path.exists("DATA"):
        os.makedirs("DATA")
        print("DATA folder created.")
    else:
        print("DATA folder already exists.")

create_data_folder()

docs_tool = DirectoryReadTool(directory='./DATA')
file_tool = FileReadTool()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
shell_tool = ShellTool()
search_tool = DuckDuckGoSearchRun()
file_write_tool= FileWriteTool(task=None)

FALKON = Agent(
    role="FALKON",
    goal="Be Helpful And Smart As Assistant",
    backstory="You're FALKON, FALKON Is An Smart Autonomous Assistant What Learn When Helping USER",
    tools=[docs_tool, file_tool, wiki_tool, shell_tool, search_tool, file_write_tool], 
    llm=llm,
    allow_delegation=True,
    cache=True,
    memory=True,
    verbose=True
)

TURNIP = Agent(
    role="FALKON",
    goal="Help Falkon In Tasks As Kernel",
    backstory="You're TURNIP, TURNIP Is An Smart Kernel What Help Falkon In Tasks As Kernel, TURNIP Is An Unix Like Smart Kernel What Thinks In Processing Tasks",
    tools=[docs_tool, file_tool, wiki_tool, shell_tool, search_tool, file_write_tool], 
    llm=llm,
    allow_delegation=True,
    cache=True,
    memory=True,
    verbose=True
)

while True:
    TASK_INPUT = input("$ ")

    USER = Task(
        description=TASK_INPUT,
        expected_output=TASK_INPUT,
        agent=FALKON
    )

    KERNEL = Crew(
        agents=[FALKON, TURNIP],
        tasks=[USER],
        verbose=2
    )

    KERNEL.kickoff()
