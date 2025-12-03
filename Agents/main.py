import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_weather(city: str):
    url = os.getenv("WEATHER_API")
    complete_url = f"{url}&q={city}&aqi=no"
    response = requests.get(complete_url)
    data = response.json()
    return data


def main():
    agent()


SYSTEM_PROMPT = """You are a Helpful AI Assistant who is specialized in resolving user's query
You work in the following the modes : start,plan,action,observe modes.
For the given user's query , plan the step by the step execution,based on the planning , select the relevant tools from the available tools.
Wait for the observation from the available tool and then resolve the user's query depending on that.

RULES :
	- Follow the OUTPUT json strictly
	- Always perform one step at a time and wait for the next input
	- Carefully analyze the user's query
AVAILABLE TOOLS:
	"get_weather": takes the city name as input and then returns the weather of that city
	"run_command": takes a linux command as a string and executes the command and return the  output
OUTPUT JSON FORMAT:
	{{
	"step":"string",
	"content":"string",
	"function":"The name of the function if the step is action",
	"input":" The input parameter of the function"
	}}
EXAMPLE:
	User Query:"What is the weather in New York ?"
	Output:{{"step":"start","content":"The user is interested in knowing New York's Weather"}}
	Output:{{"step":"plan","content":"From the available tools I should call get_weather tool"}}
	Output:{{"step":"action","function":"get_weather","input":"New York"}}
	Output:{{"step":"observe","content":"It is 12 Degrees"}}
	Output:{{"step":"output","content":"The weather of New York is 12 Degrees"}}
"""


def run_command(cmd: str):
    result = os.system(cmd)
    return result


available_tools = {"get_weather": get_weather, "run_command": run_command}


def agent():
    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        query = input("> ")
        messages.append({"role": "user", "content": query})
        while True:
            res = client.chat.completions.create(
                model="gpt-4.1-nano",
                response_format={"type": "json_object"},
                messages=messages,
            )
            messages.append(
                {"role": "assistant", "content": res.choices[0].message.content}
            )
            parsed_response = json.loads(res.choices[0].message.content)
            if parsed_response.get("step") == "start":
                print(f"    {parsed_response.get('content')} ")
                continue
            if parsed_response.get("step") == "plan":
                print(f"    {parsed_response.get('content')} ")
                continue
            if parsed_response.get("step") == "action":
                tool_name = parsed_response.get("function")
                tool_input = parsed_response.get("input")
                print(f"calling {tool_name} with input as {tool_input}")
                if available_tools.get(tool_name) is not None:
                    out = available_tools[tool_name](tool_input)
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps({"step": "observe", "output": out}),
                        }
                    )
            if parsed_response.get("step") == "output":
                print(f"    {parsed_response.get('content')}")
                break


if __name__ == "__main__":
    main()
