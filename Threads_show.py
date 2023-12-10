# websiteに表示されるアシスタントIDとスレッドIDは違うらしく、これは動かない...
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

args = sys.argv
thread_id = args[1]
thread_messages = client.beta.threads.messages.list(thread_id)

for m in thread_messages.data:
  print(m.content[0].text.value)