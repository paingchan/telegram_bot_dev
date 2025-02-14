import os
import logging
from dotenv import load_dotenv
from qdrant_operations import search_collection, search_collection_async, create_collection_if_not_exists
from s3_operations import initialize_s3_client, check_do_spaces_connection
from sentence_transformers import SentenceTransformer
import requests
import json
from config import PROMPT_TEMPLATES, ALLOWED_USERS
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
from qdrant_client import QdrantClient
from utils import get_model, get_qdrant_client
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor

# Force reload environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DO_SPACES_KEY = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION")
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET")
DO_SPACES_ENDPOINT = os.getenv("DO_SPACES_ENDPOINT")

# Embedding cache
# embedding_cache = {}
# MAX_CACHE_SIZE = 1000  # Adjust based on your needs

s3_client = boto3.client('s3',
    endpoint_url=DO_SPACES_ENDPOINT,
    aws_access_key_id=DO_SPACES_KEY,
    aws_secret_access_key=DO_SPACES_SECRET,
    config=Config(signature_version='s3v4'),
    region_name=DO_SPACES_REGION
)

def get_embedding(query):
    """Generate embedding without caching"""
    model = get_model()
    return model.encode(query, convert_to_tensor=False).tolist()

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

# Load environment variables for API keys
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Validate environment variables
if not all([DEEPSEEK_URL, DEEPSEEK_API_KEY, QDRANT_API_KEY, QDRANT_URL, TELEGRAM_BOT_TOKEN]):
    raise ValueError("Missing required environment variables. Please check your configuration.")

def call_deepseek_api(prompt):
    # Add caching logic
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    
    if False:
        logger.info("Using cached response")
        return {
            "response": "Cached response",
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
    """Save chat history to Digital Ocean Spaces"""
    try:
        # Try to get existing chat history from Spaces
        try:
            response = s3_client.get_object(
                Bucket=DO_SPACES_BUCKET,
                Key='datasets/chat_history.json'
            )
            chat_history = json.loads(response['Body'].read().decode('utf-8'))
            logger.info("Successfully retrieved existing chat_history.json")
        except s3_client.exceptions.NoSuchKey:
            logger.info("No existing chat_history.json, creating new file")
            chat_history = []
        except Exception as e:
            logger.error(f"Error reading chat_history.json: {str(e)}")
            return False

        # Create a new chat history entry matching your format
        chat_history_entry = {
            "id": len(chat_history) + 1,
            "user_ask": user_ask,    # Changed to match your format
            "respon": respon,        # Changed to match your format
            "state": state
        }

        # Append the new entry
        chat_history.append(chat_history_entry)

        # Upload updated chat history back to Spaces
        try:
            s3_client.put_object(
                Bucket=DO_SPACES_BUCKET,
                Key='datasets/chat_history.json',
                Body=json.dumps(chat_history, ensure_ascii=False, indent=4).encode('utf-8'),
                ContentType='application/json'
            )
            logger.info("Successfully saved to chat_history.json")
            return True
        except Exception as e:
            logger.error(f"Error saving chat_history.json: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Error in save_to_chat_history: {str(e)}")
        return False

def save_feedback(question, ai_response, is_correct, user_correction=None):
    """Save feedback with chat history update"""
    try:
        # Save to train_message.json
        feedback = {
            "question": question,
            "ai_response": ai_response,
            "is_correct": is_correct,
            "user_correction": user_correction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            response = s3_client.get_object(
                Bucket=DO_SPACES_BUCKET,
                Key='datasets/train_message.json'
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info("Successfully retrieved existing train_message.json")
        except s3_client.exceptions.NoSuchKey:
            logger.info("No existing train_message.json, creating new file")
            data = []
        except Exception as e:
            logger.error(f"Error reading train_message.json: {str(e)}")
            return False

        # Append new feedback
        data.append(feedback)

        # Save updated train_message.json
        try:
            s3_client.put_object(
                Bucket=DO_SPACES_BUCKET,
                Key='datasets/train_message.json',
                Body=json.dumps(data, ensure_ascii=False, indent=4).encode('utf-8'),
                ContentType='application/json'
            )
            logger.info("Successfully saved to train_message.json")
            return True
        except Exception as e:
            logger.error(f"Error saving train_message.json: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Error in save_feedback: {str(e)}")
        return False


async def check_user(update: Update) -> bool:
    """Check if the user is allowed to use the bot"""
    username = update.effective_user.username
    # Remove @ symbol if present in the username
    username = username.replace('@', '') if username else ''
    
    # Check if username is in the allowed users list (case-insensitive)
    allowed = any(username.lower() == allowed_user.replace('@', '').lower() 
                 for allowed_user in ALLOWED_USERS)
    
    if not allowed:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return False
    return True

# Telegram bot handlers
async def start(update: Update, context: CallbackContext):
    if not await check_user(update):
        return
    await update.message.reply_text("Hello! I'm your Thai language learning assistant. How can I help you today?")

# Adjust these constants at the top of the file
MAX_CONCURRENT_REQUESTS = 3  # Adjust based on your server capacity
REQUEST_TIMEOUT = 60.0  # Increased timeout for API requests
PROCESS_TIMEOUT = 90.0  # Increased overall process timeout

# Create a semaphore to limit concurrent requests
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

async def process_query_background(update: Update, query: str, context: CallbackContext):
    """Process query with improved concurrency handling"""
    async with semaphore:
        try:
            logger.info(f"Processing query: {query}")
            
            # Generate embedding for the query
            try:
                query_embedding = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, get_embedding, query
                )
                logger.info("Generated embedding successfully")
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise

            # Run searches concurrently
            try:
                knowledge_base_results = await search_collection_async("knowledge_base", query_embedding)
                chat_history_results = await search_collection_async("chat_history", query_embedding)
                logger.info(f"Search results - KB: {len(knowledge_base_results)}, CH: {len(chat_history_results)}")
            except Exception as e:
                logger.error(f"Error during search: {str(e)}")
                raise

            # Prepare match results
            match_results = []
            
            if knowledge_base_results and len(knowledge_base_results) > 0 and knowledge_base_results[0].get('score', 0) >= 0.6:
                match_results.append({
                    "question": knowledge_base_results[0]['payload']['question'],
                    "answer": knowledge_base_results[0]['payload']['answer'],
                    "score": knowledge_base_results[0]['score'],
                    "source": "knowledge_base",
                })

            if chat_history_results and len(chat_history_results) > 0 and chat_history_results[0].get('score', 0) >= 0.6:
                match_results.append({
                    "question": chat_history_results[0]['payload']['question'],
                    "answer": chat_history_results[0]['payload']['answer'],
                    "score": chat_history_results[0]['score'],
                    "source": "chat_history",
                })

            if not match_results:
                match_results.append({
                    "question": query,
                    "answer": "No matching results found.",
                    "score": 0,
                    "source": "none",
                })

            # Generate context string
            context_str = "\n\n".join(
                f"Source: {result['source']}\nAnswer: {result['answer']}"
                for result in match_results
                if result["score"] >= 0.6
            )

            # Generate prompt and get AI response
            try:
                prompt = generate_prompt(context_str, query)
                logger.info(f"Generated prompt: {prompt}")
                
                ai_response_data = await call_deepseek_api_async(prompt)
                logger.info(f"Got AI response: {ai_response_data}")
                
                return ai_response_data
            except Exception as e:
                logger.error(f"Error in API call: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in process_query_background: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "response": f"Sorry, there was an error processing your request: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "use_cache": False,
            }

@retry(
    stop=stop_after_attempt(2),  # Reduced retry attempts
    wait=wait_exponential(multiplier=1, min=2, max=4)  # Shorter wait times
)
async def call_deepseek_api_with_retry(client, url, headers, payload, timeout):
    response = await client.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    return response

async def call_deepseek_api_async(prompt):
    """Asynchronous version of call_deepseek_api"""
    try:
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
            "temperature": 0.3,
            "max_tokens": 200
        }

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            try:
                response = await call_deepseek_api_with_retry(
                    client,
                    DEEPSEEK_URL,
                    headers,
                    payload,
                    timeout=REQUEST_TIMEOUT
                )
                
                response_data = response.json()
                ai_response = response_data["choices"][0]["message"]["content"]
                
                return {
                    "response": ai_response,
                    "input_tokens": response_data["usage"]["prompt_tokens"],
                    "output_tokens": response_data["usage"]["completion_tokens"],
                    "total_tokens": response_data["usage"]["total_tokens"],
                    "use_cache": False,
                }
                
            except Exception as e:
                logger.error(f"Error in API request: {str(e)}")
                logger.exception("Full traceback:")
                raise
                
    except Exception as e:
        logger.error(f"Error in call_deepseek_api_async: {str(e)}")
        logger.exception("Full traceback:")
        return {
            "response": "Sorry, the service is currently busy. Please try again in a moment.",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "use_cache": False,
        }

async def handle_user_correction(update: Update, context: CallbackContext):
    """Handle user correction and save to chat history"""
    if "awaiting_correction" in context.user_data:
        user_correction = update.message.text
        correction_data = context.user_data["awaiting_correction"]
        user_query = correction_data["user_query"]
        ai_response = correction_data["ai_response"]

        # Save the feedback with the correction
        save_feedback(user_query, ai_response, is_correct=False, user_correction=user_correction)
        
        # Save the corrected answer to chat history
        save_to_chat_history(user_query, user_correction, state=1)
        
        await update.message.reply_text("ကျေးဇူးတင်ပါတယ်! အဖြေမှန်ကို မှတ်သားထားလိုက်ပါပြီ။")

        # Clear the correction context
        del context.user_data["awaiting_correction"]
    else:
        # If not awaiting correction, treat as new query
        await handle_message(update, context)

async def handle_button_click(update: Update, context: CallbackContext):
    """Handle button clicks with improved error handling and logging"""
    try:
        query = update.callback_query
        logger.info(f"Received callback query: {query.data}")
        
        await query.answer()  # Acknowledge the button click

        # Get the stored context
        last_interaction = context.user_data.get("last_interaction", {})
        user_query = last_interaction.get("user_query")
        ai_response = last_interaction.get("ai_response")

        if not user_query or not ai_response:
            logger.error("Missing context data in button handler")
            await query.edit_message_text(
                text="Error: Context missing. Please ask your question again."
            )
            return

        if query.data == "correct":
            logger.info("Processing 'correct' feedback")
            # Save to feedback and chat history
            save_feedback(user_query, ai_response, is_correct=True)
            save_to_chat_history(user_query, ai_response, state=1)
            
            await query.edit_message_text(
                text=f"{ai_response}\n\n✅ ကျေးဇူးတင်ပါတယ်! အဖြေမှန်ကို မှတ်သားထားလိုက်ပါပြီ။"
            )
        elif query.data == "incorrect":
            logger.info("Processing 'incorrect' feedback")
            await query.edit_message_text(
                text=f"AI ရဲ့အဖြေက: {ai_response}\n\nအဖြေမှန်ရေးပေးပါနော်:"
            )
            # Store the context for correction
            context.user_data["awaiting_correction"] = {
                "user_query": user_query,
                "ai_response": ai_response
            }
        else:
            logger.warning(f"Unknown callback data received: {query.data}")

    except Exception as e:
        logger.error(f"Error in handle_button_click: {str(e)}")
        logger.exception("Full traceback:")
        try:
            await query.edit_message_text(
                text="Sorry, an error occurred while processing your feedback."
            )
        except Exception as sub_e:
            logger.error(f"Error sending error message: {sub_e}")

async def handle_message(update: Update, context: CallbackContext):
    if not await check_user(update):
        return

    # Check if user typed the confirmation text
    if update.message.text == "✅ မှန်တယ်":
        if "last_interaction" in context.user_data:
            last_interaction = context.user_data["last_interaction"]
            user_query = last_interaction.get("user_query")
            ai_response = last_interaction.get("ai_response")
            
            if user_query and ai_response:
                # Save as correct feedback and to chat history
                save_feedback(user_query, ai_response, is_correct=True)
                save_to_chat_history(user_query, ai_response, state=1)
                await update.message.reply_text("ကျေးဇူးတင်ပါတယ်! အဖြေမှန်ကို မှတ်သားထားလိုက်ပါပြီ။")
                return
            else:
                await update.message.reply_text("ဝမ်းနည်းပါတယ်။ ပြီးခဲ့တဲ့မေးခွန်းကို ရှာမတွေ့ပါဘူး။")
                return

    if "awaiting_correction" in context.user_data:
        await handle_user_correction(update, context)
        return

    query = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    logger.info(f"Received message from user {user_id}: {query}")

    # More informative processing message
    processing_msg = await update.message.reply_text(
        "ခဏစောင့်ပေးပါ... ကျွန်မ အဖြေရှာပေးနေပါတယ်။\n"
        "(Processing your request, please wait...)"
    )

    try:
        response_data = await asyncio.wait_for(
            process_query_background(update, query, context),
            timeout=PROCESS_TIMEOUT
        )
        
        # Store the context for the callback
        context.user_data["last_interaction"] = {
            "user_query": query,
            "ai_response": response_data["response"]
        }

        # Create the keyboard markup
        keyboard = [
            [
                InlineKeyboardButton("✅ မှန်တယ်", callback_data="correct"),
                InlineKeyboardButton("❌ မှားတယ်", callback_data="incorrect")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send the response with buttons
        await update.message.reply_text(
            text=response_data["response"],
            reply_markup=reply_markup
        )
        logger.info(f"Sent response to user {user_id}")

    except asyncio.TimeoutError:
        await update.message.reply_text(
            "ဝမ်းနည်းပါတယ်။ အချိန်ကြာသွားလို့ပါ။\n"
            "ခဏလေးနေ ပြန်မေးကြည့်ပါ။"
        )
    except Exception as e:
        logger.error(f"Error processing message from user {user_id}: {str(e)}")
        await update.message.reply_text(
            "ဝမ်းနည်းပါတယ်။ error ဖြစ်နေပါတယ်။\n"
            "ခဏလေးနေ ပြန်မေးကြည့်ပါ။"
        )
    finally:
        # Delete processing message
        try:
            await processing_msg.delete()
        except Exception as e:
            logger.error(f"Error deleting processing message: {str(e)}")

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
    # Configure logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize S3 client
    s3_client = initialize_s3_client()
    
    # Check DO Spaces connection
    if not s3_client or not check_do_spaces_connection(s3_client):
        logger.error("Failed to connect to Digital Ocean Spaces. Check your credentials and connection.")
        raise SystemExit("Cannot continue without Digital Ocean Spaces connection")

    # Check collections without recreating them
    try:
        create_collection_if_not_exists("knowledge_base")
        create_collection_if_not_exists("chat_history")
    except Exception as e:
        logger.error(f"Error checking collections: {e}")

    # Start the Telegram bot with concurrent updates
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)  # Enable concurrent updates
        .build()
    )

    # Add handlers in the correct order
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_button_click))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(InlineQueryHandler(inline_query))

    # Start the bot with all update types enabled
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)