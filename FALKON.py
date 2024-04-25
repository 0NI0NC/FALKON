### IMPORTS ###
import os
import getpass
from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ShellTool, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_google_genai import ChatGoogleGenerativeAI

### MODEL ###
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY: NOT FOUND")

### ASCII ###
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

### TOOL ###
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

### DATA ###
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

### TOOLS ###
docs_tool = DirectoryReadTool(directory='./DATA')
file_tool = FileReadTool()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
shell_tool = ShellTool()
search_tool = DuckDuckGoSearchRun()
file_write_tool= FileWriteTool(task=None)

### AGENTS ###
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
    role="TURNIP", 
    goal="Help Falkon In Tasks As Kernel",
    backstory="You're TURNIP, TURNIP Is An Smart Kernel What Help Falkon In Tasks As Kernel, TURNIP Is An Unix Like Smart Kernel What Thinks In Processing Tasks",
    tools=[docs_tool, file_tool, wiki_tool, shell_tool, search_tool, file_write_tool], 
    llm=llm,
    allow_delegation=True, 
    cache=True,
    memory=True,
    verbose=True
)

### LOOP ###
while True:
    TASK_INPUT = input("$ ")

    USER = Task(
        description=TASK_INPUT,
        expected_output=TASK_INPUT,
        agent=FALKON,
        allow_delegation=True,
        llm=llm,
    )

    KERNEL = Crew(
        agents=[FALKON, TURNIP],
        tasks=[USER],
        verbose=2,
        allow_delegation=True
    )

    KERNEL.kickoff()
