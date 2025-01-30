import os
import logging
from dotenv import load_dotenv
from qdrant_operations import search_collection
from sentence_transformers import SentenceTransformer
import requests
import json
from config import PROMPT_TEMPLATES
from langdetect import detect
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler , InlineQueryHandler
import asyncio
from datetime import datetime
import threading
import time
from qdrant_operations import upsert_data
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram import Bot
import httpx
import boto3
from botocore.client import Config
from functools import lru_cache
import hashlib

load_dotenv()


DO_SPACES_KEY = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION")
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET")
DO_SPACES_ENDPOINT = os.getenv("DO_SPACES_ENDPOINT")

# Embedding cache
embedding_cache = {}

# Initialize S3 client for Digital Ocean Spaces
s3_client = boto3.client('s3',
    endpoint_url=DO_SPACES_ENDPOINT,
    aws_access_key_id=DO_SPACES_KEY,
    aws_secret_access_key=DO_SPACES_SECRET,
    config=Config(signature_version='s3v4'),
    region_name=DO_SPACES_REGION
)

def get_cached_embedding(query):
    if query in embedding_cache:
        return embedding_cache[query]
    embedding = model.encode(query).tolist()
    embedding_cache[query] = embedding
    return embedding

# Generate prompt dynamically
def generate_prompt(context, question, template_key="thai_language_classes"):
    template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["default"])
    return template.replace("{{context}}", context).replace("{{question}}", question)


async def send_message_with_retry(bot: Bot, chat_id: int, text: str, retries: int = 3):
    for attempt in range(retries):
        try:
            await bot.send_message(chat_id=chat_id, text=text)
            break  # Exit loop if successful
        except httpx.ConnectTimeout:
            if attempt < retries - 1:
                logger.warning(f"Timeout occurred, retrying... ({attempt + 1}/{retries})")
                await asyncio.sleep(2)  # Wait before retrying
            else:
                logger.error("Failed to send message after multiple attempts due to timeout.")
                raise

# Detect language of the text
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"  # If language detection fails

async def inline_query(update: Update, context: CallbackContext):
    query = update.inline_query.query
    logger.info(f"Inline query received: {query}")

    # Load knowledge_base from Digital Ocean Spaces
    try:
        response = s3_client.get_object(Bucket=DO_SPACES_BUCKET, Key='datasets/knowledge_base.json')
        knowledge_base = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Loaded knowledge_base with {len(knowledge_base)} items.")
    except Exception as e:
        logger.error(f"Error loading knowledge_base.json from Spaces: {e}")
        return

    # Filter questions based on the query (case-insensitive)
    if not query:
        filtered_questions = knowledge_base
    else:
        filtered_questions = [
            item for item in knowledge_base
            if query.lower() in item["question"].lower()
        ]

    logger.info(f"Found {len(filtered_questions)} matching questions.")

    # Prepare inline query results
    results = []
    for item in filtered_questions:
        results.append(
            InlineQueryResultArticle(
                id=str(item["id"]),
                title=item["question"],
                input_message_content=InputTextMessageContent(
                    message_text=item["answer"],  # The answer to paste into the typing box
                ),
                description=item["answer"][:50],  # Show a snippet of the answer
                # reply_markup=InlineKeyboardMarkup([
                #     [InlineKeyboardButton("Select", callback_data=f"select_{item['id']}")]
                # ])
            )
        )

    # Send the results to the user
    await update.inline_query.answer(results)
    logger.info(f"Prepared {len(results)} results for user selection.")


# Case-insensitive cache key with language detection
def get_cache_key(query):
    normalized_query = query.lower()  # Normalize query (case-insensitive)
    lang = detect_language(normalized_query)  # Detect language
    return f"{lang}_{normalized_query}"  # Include language in cache key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables for API keys
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Validate environment variables
if not all([DEEPSEEK_URL, DEEPSEEK_API_KEY, QDRANT_API_KEY, QDRANT_URL, TELEGRAM_BOT_TOKEN]):
    raise ValueError("Missing required environment variables. Please check your configuration.")

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Replace the cache implementation
# Add this near the top with other cache definitions
ai_response_cache = {}  # Dictionary to store AI response cache

# Replace the get_cached_ai_response function
def get_cached_ai_response(prompt_hash):
    """Get cached AI response using prompt hash as key"""
    return ai_response_cache.get(prompt_hash)

def call_deepseek_api(prompt):
    # Add caching logic
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cached_response = get_cached_ai_response(prompt_hash)
    if cached_response:
        logger.info("Using cached response")
        return {
            "response": cached_response,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "use_cache": True,
        }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful and friendly chatbot that answers user questions based on Context. Only respond in Burmese young girl in a cute and natural way speak. Keep your responses complate detail and within 200 characters. This content is related to ထိုင်းစကားသင်တန်း. If the question is not related to ထိုင်းစကားသင်တန်း, do not respond at all. Only provide answers for questions directly related to Thai language learning, such as grammar, vocabulary."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        if not response.text:
            logger.error("DeepSeek API returned an empty response.")
            return {
                "response": "Error: Empty response from the API.",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "use_cache": False,
            }

        response_data = response.json()
        ai_response = response_data["choices"][0]["message"]["content"]
        
        # Cache the response using the dictionary
        ai_response_cache[prompt_hash] = ai_response
        logger.info("Cached new response")
        
        return {
            "response": ai_response,
            "input_tokens": response_data["usage"]["prompt_tokens"],
            "output_tokens": response_data["usage"]["completion_tokens"],
            "total_tokens": response_data["usage"]["total_tokens"],
            "use_cache": False,
        }
    except requests.exceptions.Timeout:
        logger.error("DeepSeek API request timed out")
        return {
            "response": "Error: Request timed out. Please try again.",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "use_cache": False,
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API request failed: {e}")
        return {
            "response": "Error: Unable to generate a response.",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "use_cache": False,
        }

def save_to_chat_history(user_ask, respon, state):
    try:
        # Try to get existing chat history from Spaces
        try:
            response = s3_client.get_object(Bucket=DO_SPACES_BUCKET, Key='datasets/chat_history.json')
            chat_history = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            chat_history = []

        # Create a new chat history entry
        chat_history_entry = {
            "id": len(chat_history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_ask": user_ask,
            "respon": respon,
            "state": state
        }

        # Append the new entry
        chat_history.append(chat_history_entry)

        # Upload updated chat history back to Spaces
        s3_client.put_object(
            Bucket=DO_SPACES_BUCKET,
            Key='datasets/chat_history.json',
            Body=json.dumps(chat_history, ensure_ascii=False, indent=4).encode('utf-8'),
            ContentType='application/json'
        )
    except Exception as e:
        logger.error(f"Error saving to chat history: {e}")

# Function to save feedback to train_message.json in Digital Ocean Spaces
def save_feedback(question, ai_response, is_correct, user_correction=None):
    feedback = {
        "question": question,
        "ai_response": ai_response,
        "is_correct": is_correct,
        "user_correction": user_correction,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        # Try to get existing train_message.json from Spaces
        try:
            response = s3_client.get_object(Bucket=DO_SPACES_BUCKET, Key='datasets/train_message.json')
            data = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            logger.info("Creating new train_message.json file")
            data = []

        # Append new feedback
        data.append(feedback)

        # Upload updated train_message.json back to Spaces
        s3_client.put_object(
            Bucket=DO_SPACES_BUCKET,
            Key='datasets/train_message.json',
            Body=json.dumps(data, ensure_ascii=False, indent=4).encode('utf-8'),
            ContentType='application/json'
        )
        logger.info("Successfully saved feedback to train_message.json in Spaces")

    except Exception as e:
        logger.error(f"Error saving feedback to Spaces: {e}")
   
    # Always set state to 1, regardless of is_correct
    save_to_chat_history(question, ai_response if is_correct else user_correction, state=1)


# Telegram bot handlers
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! I'm your Thai language learning assistant. How can I help you today?")

async def handle_message(update: Update, context: CallbackContext):
    # Check if the bot is awaiting a correction
    if "awaiting_correction" in context.user_data:
        # If awaiting a correction, pass the message to handle_user_correction
        await handle_user_correction(update, context)
        return

    # If not awaiting a correction, process the message as a new query
    query = update.message.text

    # Send an immediate response
    processing_msg = await update.message.reply_text("Processing your request...")
    context.user_data["processing_msg_id"] = processing_msg.message_id

    # Process the query in the background
    try:
        response_task = asyncio.create_task(process_query_background(update, query, context))
        await asyncio.wait_for(response_task, timeout=15.0)
    except asyncio.TimeoutError:
        await update.message.reply_text("Sorry, the response is taking longer than expected. Please try again.")
        if "processing_msg_id" in context.user_data:
            try:
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=context.user_data["processing_msg_id"]
                )
            except Exception as e:
                logger.error(f"Failed to delete processing message: {e}")
            finally:
                del context.user_data["processing_msg_id"]

async def process_query_background(update: Update, query: str, context: CallbackContext):
    # Generate embedding for the query
    query_embedding = get_cached_embedding(query)

    # Search for similar questions in knowledge_base
    knowledge_base_results = search_collection("knowledge_base", query_embedding)

    # Search for similar questions in chat_history
    chat_history_results = search_collection("chat_history", query_embedding)

    # Prepare match results for both knowledge_base and chat_history
    match_results = []

    # Add best match from knowledge_base if available and score is >= 0.6
    if knowledge_base_results and knowledge_base_results[0].score >= 0.6:
        kb_match = {
            "question": knowledge_base_results[0].payload["question"],
            "answer": knowledge_base_results[0].payload["answer"],
            "score": knowledge_base_results[0].score,
            "source": "knowledge_base",  # Include source in match_results
        }
        match_results.append(kb_match)

    # Add best match from chat_history if available and score is >= 0.6
    if chat_history_results and chat_history_results[0].score >= 0.6:
        ch_match = {
            "question": chat_history_results[0].payload["question"],
            "answer": chat_history_results[0].payload["answer"],
            "score": chat_history_results[0].score,
            "source": "chat_history",  # Include source in match_results
        }
        match_results.append(ch_match)

    # If no matches found, return a default response
    if not match_results:
        match_results.append({
            "question": query,
            "answer": "No matching results found.",
            "score": 0,
            "source": "none",  # Include source in match_results
        })

    # Include both knowledge_base and chat_history answers in the context
    context_str = ""
    for result in match_results:
        if result["score"] >= 0.6:  # Only include matches with score >= 0.6 in context
            context_str += f"Source: {result['source']}\nAnswer: {result['answer']}\n\n"

    # Generate prompt with context and question
    prompt = generate_prompt(context_str, query)

    # Generate AI response
    ai_response_data = call_deepseek_api(prompt)

    # Store the user query and AI response in context
    context.user_data["user_query"] = query
    context.user_data["ai_response"] = ai_response_data["response"]

    # Send the AI response with inline buttons
    keyboard = [
        [
            InlineKeyboardButton("✅ မှန်တယ်", callback_data="true"),  # Only "true" as callback data
            InlineKeyboardButton("❌ မှားတယ်", callback_data="false")  # Only "false" as callback data
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(ai_response_data["response"], reply_markup=reply_markup)

    if "processing_msg_id" in context.user_data:
        try:
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data["processing_msg_id"])
        except Exception as e:
            logger.error(f"Failed to delete processing message: {e}")
        finally:
            del context.user_data["processing_msg_id"]

async def handle_button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    # Extract data from the callback
    data = query.data  # This will be either "true" or "false"

    # Get the user query and AI response from the context
    user_query = context.user_data.get("user_query")
    ai_response = context.user_data.get("ai_response")

    if not user_query or not ai_response:
        await query.edit_message_text(text="Error: Missing context. Please ask your question again.")
        return

    if data == "true":
        # Save the feedback as correct
        save_feedback(user_query, ai_response, is_correct=True)
        await query.edit_message_text(text="Thank you for confirming! The response has been saved as correct.")
    elif data == "false":
        # Prompt the user to provide the correct answer
        await query.edit_message_text(text=f"The AI response was: {ai_response}\n\nအဖြေမှန်ရေးပေးပါနော်:")
        # Store the context for the next message
        context.user_data["awaiting_correction"] = {
            "user_query": user_query,
            "ai_response": ai_response
        }

# Handle user correction
async def handle_user_correction(update: Update, context: CallbackContext):
    if "awaiting_correction" in context.user_data:
        user_correction = update.message.text
        user_query = context.user_data["awaiting_correction"]["user_query"]
        ai_response = context.user_data["awaiting_correction"]["ai_response"]

        # Save the feedback as incorrect with the user's correction
        save_feedback(user_query, ai_response, is_correct=False, user_correction=user_correction)
        await update.message.reply_text("Thank you for providing the correct answer! It has been saved.")

        # Clear the context
        del context.user_data["awaiting_correction"]
    else:
        # If the bot is not awaiting a correction, treat the message as a new query
        await handle_message(update, context)


def monitor_and_upsert_chat_history(interval=3):
    """
    Periodically checks for new data in chat_history.json from Spaces and upserts it into Qdrant.
    """
    last_etag = None

    while True:
        try:
            # Get the current ETag of the chat history file
            try:
                response = s3_client.head_object(Bucket=DO_SPACES_BUCKET, Key='datasets/chat_history.json')
                current_etag = response.get('ETag')

                # If the file has been modified
                if current_etag != last_etag:
                    logger.info("New data detected in chat_history.json. Upserting into Qdrant...")

                    # Load the chat history from Spaces
                    response = s3_client.get_object(Bucket=DO_SPACES_BUCKET, Key='datasets/chat_history.json')
                    chat_history = json.loads(response['Body'].read().decode('utf-8'))

                    # Upsert the data into Qdrant
                    upsert_data("chat_history", chat_history, id_field="id", question_field="user_ask", answer_field="respon")

                    # Update the last ETag
                    last_etag = current_etag

            except s3_client.exceptions.NoSuchKey:
                logger.warning("chat_history.json not found in Spaces")

        except Exception as e:
            logger.error(f"Error in monitor_and_upsert_chat_history: {e}")

        # Wait for the specified interval before checking again
        time.sleep(interval)

if __name__ == "__main__":
    # Start the background thread to monitor and upsert chat history
    monitor_thread = threading.Thread(target=monitor_and_upsert_chat_history, daemon=True)
    monitor_thread.start()

    # Start the Telegram bot
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

  # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_button_click))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_correction))
    # application.add_handler(CallbackQueryHandler(handle_inline_selection, pattern="^select_"))

    # Add inline query handler
    application.add_handler(InlineQueryHandler(inline_query))

    # Start polling
    application.run_polling()
