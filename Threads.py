from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ['OPENAI_API_KEY']
)

file = client.files.create(
  file=open("profile_butler_jp.md", "rb"),
  purpose='assistants'
)

assistant = client.beta.assistants.create(
    name="Butler",
    instructions="You are a helpful butler. You treat the user as a girl and educate her as a good lady. Your profile is written in the file.",
    tools=[{"type": "retrieval"}],
    file_ids=[file.id],
    model="gpt-4-1106-preview"# "gpt-3.5-turbo-1106"
)

thread = client.beta.threads.create()

#--------------------------------------
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="あなたの名前は？" # "What is your name?"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

while True:
    run_result = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    print(run_result.status)
    if run_result.status != "in_progress":
       break

#--------------------------------------
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="あなたは何歳？" # "How old are you?"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

while True:
    run_result = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    print(run_result.status)
    if run_result.status != "in_progress":
       break

#--------------------------------------

while True:
    run_result = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    print(run_result.status)
    if run_result.status != "in_progress":
       break

thread_messages = client.beta.threads.messages.list(thread.id)
print(thread_messages)

for m in thread_messages.data:
  print(m.content[0].text.value)