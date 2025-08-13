from dotenv import load_dotenv
from typing import Any
from langchain import hub
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent
from langchain.tools import Tool

def main():
    print("Start...")

    instructions = """
    You are an agent designed to write and execute python code to answer questions.
    You have access to a Python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only  use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer 
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")

    prompt  = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    python_agent = create_react_agent(
        llm=ChatOllama(model="qwen3:1.7b", temperature=0),
        tools=tools,
        prompt=prompt
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True, max_iterations=5, handle_parsing_errors=True)

    # python_agent_executor.invoke(input={"input": "generate and save in current working directory 15 QR codes that point to www.udemy.com/course/langchain, you have qrcode package installed already"})

    csv_agent_executor = create_csv_agent(llm=ChatOllama(model="qwen3:1.7b", temperature=0), path="episode_info.csv", allow_dangerous_code=True, verbose=True)

    # csv_agent_executor.invoke(input={"input": "how many columns are there in the episode_info.csv file?"})

    # csv_agent_executor.invoke(input={"input": "which writer wrote the most episodes? How many episodes did he write?"})

    # csv_agent_executor.run("print seasons in ascending order of the number of episodes they have")

    ########################## Router Grannd Agent ##########################

    def python_agent_executor_wrapper(original_prompt:str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(name="Python Agent",
        func=python_agent_executor_wrapper,
        description="""
        useful when you need to transform natural language to python and execute the python code, returning the results of the code execution.
        DOES NOT ACCEPT CODE AS INPUT!
        """
        ),
        Tool(name="CSV Agent",
        func=csv_agent_executor.invoke,
        description="""
        useful when you need to answer questions using episode_info.csv file, takes the entire question as an input and returns the answer after running the pandas calculations.
        """
        )
    ]

    prompt = base_prompt.partial(instructions="")

    grand_agent = create_react_agent(prompt=prompt, llm=ChatOllama(model="qwen3:4b", temperature=0), tools=tools)

    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(grand_agent_executor.invoke(input={"input": "which season has the most episodes?"}))
    print(grand_agent_executor.invoke(input={"input": "Generate and save in current working directory 15 qr codes that point to www.udemy.com/course/langchain"}))

if __name__  == "__main__":
    main()