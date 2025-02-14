# config.py
PROMPT_TEMPLATES = {
    "thai_language_classes": '''
    User Question:
    {{question}}

    Context:
    {{context}}
    ''',
    "default": '''
    User Question:
    {{question}}
    
    Context:
    {{context}}
    '''
}

# Define testers
USER_PERMISSIONS = {
    "testers": [
        "blackghost000",
        "paingchan009",
        "tester2",
        "tester3",
        "tester4",
        "tester5"
        # Add more testers as needed
    ]
}

# Use only testers as allowed users
ALLOWED_USERS = USER_PERMISSIONS["testers"]