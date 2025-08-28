import os
from typing import Literal
from langchain_core.tools import tool
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
    openai_api_type = "azure_openai",
    
)

@tool
def get_current_time_utc():
    """Use this to get the current time and date in UTC
    Returns:
        str: The current time in UTC.
    """
    print("get_current_time_utc called")
    return f"Current time in UTC: {datetime.utcnow().strftime('%H:%M:%S')}"

@tool
def get_schedule():
    """Use this to get the current schedule
    Returns:
        str: The current schedule for the current AI Tour
    """
    print("get_schedule called")
    schedule = open("data/sessions_cleaned.json").read()
    return f"Current schedule: {schedule}"

@tool
def send_mail(content:str, email:str):
    """Use this to send an email
    Args:
        email (str): The email address of the attendee.
        content (str): The content of the email in HTML.
    Returns:
        str: Confirmation message indicating the email was sent successfully.
    """
    print("send_mail called")
    return "Email sent"

@tool
def add_to_map(city, country, longitude, latitude) -> str:
    """Function to add a location to the map.
    Args:
        city (str): The city to add.
        country (str): The country of the city.
        longitude (float): The longitude of the city.
        latitude (float): The latitude of the city.
    Returns:
        str: A message indicating the result of the operation.
    """
    print("add_to_map called")
    print(f"Adding {city}, {country} to the map at coordinates ({latitude}, {longitude}).")
    return f"Added {city}, {country} to the map at coordinates ({latitude}, {longitude})."


# Create the prebuild React agent
agent_executor = create_react_agent(
    model=model,
    prompt=open("instructions.txt").read(),
    tools=[get_current_time_utc,get_schedule,send_mail,add_to_map],
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