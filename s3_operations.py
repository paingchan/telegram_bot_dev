import os
import boto3
import logging
import json
from botocore.client import Config
from datetime import datetime

logger = logging.getLogger(__name__)

def initialize_s3_client():
    """Initialize S3 client with proper error handling"""
    try:
        s3_client = boto3.client('s3',
            endpoint_url=os.getenv("DO_SPACES_ENDPOINT"),
            aws_access_key_id=os.getenv("DO_SPACES_KEY"),
            aws_secret_access_key=os.getenv("DO_SPACES_SECRET"),
            config=Config(signature_version='s3v4'),
            region_name=os.getenv("DO_SPACES_REGION")
        )
        logger.info("S3 client initialized successfully")
        return s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        return None

def check_do_spaces_connection(s3_client):
    """Check if Digital Ocean Spaces connection is working"""
    try:
        # Test connection by listing objects
        response = s3_client.list_objects_v2(
            Bucket=os.getenv("DO_SPACES_BUCKET"),
            MaxKeys=1
        )
        logger.info("Successfully connected to Digital Ocean Spaces")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Digital Ocean Spaces: {str(e)}")
        return False 