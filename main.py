import os
import chainlit as cl
from dotenv import find_dotenv, load_dotenv
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from typing import List, Dict, Any

# Load environment variables
load_dotenv(find_dotenv())

# Get API key and validate
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Step 3: Config define at run level
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Step 4: Agent
agent = Agent(
    name="HealthSage",
    instructions="You are a health assistant. Provide accurate and helpful responses to health-related queries.",
)

@cl.on_chat_start
async def handle_chat_start():
    """Initialize chat session when a user connects."""
    # Initialize empty chat history
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Hello! I'm your HealthSage AI assistant. How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("chat_history")
    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        agent, 
        input=history,
        run_config=config
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("chat_history", history)
    await cl.Message(content=result.final_output).send()
