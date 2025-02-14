from huggingface_hub import InferenceClient

client = InferenceClient(
	provider="together",
	api_key="sk-c17130735c7f4539b635061e4f91f178"
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)