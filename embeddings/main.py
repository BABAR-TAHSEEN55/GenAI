import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def Embeddings():
    text = "I am human"
    client = OpenAI()
    res = client.embeddings.create(input=text, model="text-embedding-3-small")
    print(res.data[0].embedding)


# Zero Shot Prompting
SYSTEM_PROMPT = "You are the the best python teacher.Only solve python doubts.If someone asks anything else brutally roast them"
# Few Shot Prompting
SYSTEM_PROMPT2 = """You are the the best python teacher.Only solve python doubts.If someone asks anything else brutally roast them
Example
User:"Hey how are you"
Assistant:"Shut up shit"
"""


def Init():
    client = OpenAI()
    res = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT2},
            {"role": "user", "content": "Hey how are you ?"},
            {"role": "system", "content": "I'm Batman"},
            {"role": "user", "content": "How am I?"},
            # {"role": "user", "content": "Functions in python"},
        ],
    )
    print(res.choices[0].message.content)


SYSTEM_CHAIN_PROMPT = """
You are a Grammer checker AI assistant specialized in improving and correcting the user's query with high accuracy.

Your task is to refine or analyze the user input step-by-step while producing short summarized reasoning only.

Follow this step sequence exactly:
"analyze" → "think" → "output" → "validate" → "result"

Rules:
1. Return ONLY valid JSON following this exact schema:

   {"step":"<step-name>","content":"<message>"}

2. Perform ONLY one step per response and wait for the next user input before continuing.
3. Keep reasoning brief and summarized (do NOT reveal private chain-of-thought).
4. Check grammar, clarity, and correctness carefully.
5. Use the exact step names provided (case-sensitive).

Example session:

User: Chinese I am.

Assistant:
{"step":"analyze","content":"The sentence structure is reversed and needs correction."}

Assistant:
{"step":"think","content":"Reordering the words will produce a grammatically correct sentence."}

Assistant:
{"step":"output","content":"I am Chinese."}

Assistant:
{"step":"validate","content":"The corrected sentence is grammatically valid."}

Assistant:
{"step":"result","content":"The final refined query is: \"I am Chinese.\""}
"""


def ChainPrompt():
    client = OpenAI()
    res = client.chat.completions.create(
        model="gpt-4.1-nano",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_CHAIN_PROMT},
            {"role": "user", "content": "Are how you?"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "analyse",
                        "content": "The user asked 'Are how you?' which appears to be a garbled sentence. It likely intends 'How are you?'. I will interpret and refine the query to the proper question.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "think",
                        "content": "Interpret 'Are how you?' as 'How are you?' and plan to refine the query to the proper question.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "analyze",
                        "content": "The input 'Are how you?' is ungrammatical. It likely intends the friendly greeting 'How are you?'. I'll refine it to the proper question 'How are you?'.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "think again",
                        "content": "Re-evaluate the input 'Are how you?' for possible intended meaning; the most plausible refinement is 'How are you?'; plan to present this refined question in the next step.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({"step": "output", "content": "How are you?"}),
            },
        ],
    )
    print(res.choices[0].message.content)


def AutonomousBot():
    messages = [{"role": "system", "content": SYSTEM_CHAIN_PROMPT}]

    client = OpenAI()
    query = input("> ")
    messages.append({"role": "user", "content": query})

    while True:
        res = client.chat.completions.create(
            model="gpt-4.1-nano",
            response_format={"type": "json_object"},
            messages=messages,
        )
        parsed_response = json.loads(res.choices[0].message.content)
        if parsed_response.get("step") != "result":
            print("🧠: ", parsed_response.get("content"))
            continue
        print("🤖: ", parsed_response.get("content"))
        break


def main():
    # Embeddings()
    # Init()
    # ChainPrompt()
    AutonomousBot()


if __name__ == "__main__":
    main()

#
# Offshore the validation to deepseek
# Persona Maker make it sound natural 50-80exampels 200lines
