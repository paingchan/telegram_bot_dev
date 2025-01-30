# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get the actual API key from environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Pass the actual API key value, not the string "DEEPSEEK_API_KEY"
client = OpenAI(
    api_key="sk-565cdc0a39244527b371e579e828ed24",  # Remove the quotes
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # Added /v1 to the base URL
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        # {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)