from langchain_core.prompts import ChatPromptTemplate

"""
Da ka2eno message placeholder btdelo messages b roles mo5talefa
w hoa y3mel construct le el prompt fe el a5er

Hoa aktar yo3tabar template momken ykon feh amaken fadya w enta bt3mel el 
prompt btdehalo ka input

el modo3 msh bs message ba3melaha construct, la hoa ka2eno chat already sh3'al 
mabeen tlata (human/user, ai, system) w el chat history da bysa3ed el llm eno yefham el context already

human: da el human 2alo
ai: da el rad bta3 el ai 3la el human
system: da system-level instructions or guidelines that guide the behavior of the AI
"""

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?"
    }
)
print(prompt_value.to_string())