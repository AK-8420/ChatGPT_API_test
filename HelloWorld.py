from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # api_key="My API Key",
    api_key=os.environ['OPENAI_API_KEY']
)

response = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [{
        "role":"user", 
        "content":"Hello World! This is the first message I send you through API."
    }]
) 

print(response)