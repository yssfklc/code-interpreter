from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI


load_dotenv()
tools = [PythonREPLTool()]


def main():
    print("And project starts...")
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    agent = create_openai_functions_agent(ChatOpenAI(temperature=0), tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    # agent_executor.invoke(
    #     {
    #         "input": """Understand, write a nodejs backend OAuth authentication code. Which takes password and username as input and chek it whether user has authenticated."""
    #     }
    # )

    agent_csv = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        "episode_info.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    agent_csv.invoke({"input":"Which writer wrote the most episodes? How many episodes did he write"})


if __name__ == "__main__":

    main()
