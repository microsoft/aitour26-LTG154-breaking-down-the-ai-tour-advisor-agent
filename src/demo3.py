import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import chainlit as cl
from datetime import datetime

# Setup the connection to the Model in Azure AI Foundry
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

instructions = open("instructions.txt").read()+f"\n\nCurrent date: {datetime.now().strftime('%Y-%m-%d')}\nCurrent time: {datetime.now().strftime('%H:%M:%S')} in CDT"

# Create the prebuild React 
agent_executor = create_react_agent(
    model=model,
    prompt=instructions,
    tools=[],
    checkpointer=InMemorySaver()
)

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    final_answer = cl.Message(content="")
    
    for msg, metadata in agent_executor.stream({"messages": [HumanMessage(content=msg.content)]}, config, stream_mode="messages"):
        if metadata['langgraph_node'] == "agent":
            await final_answer.stream_token(msg.content)

    await final_answer.send()