from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/category_chain/")
remote_chain.invoke({
    "character": "proffesional butler",
    "user_role": "your master",
    "user_input": "あなたは力持ちですか？"
})