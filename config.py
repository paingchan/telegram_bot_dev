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