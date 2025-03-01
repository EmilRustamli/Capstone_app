from app import app

# This is the handler function Vercel will call
def handler(request, context):
    return app(request, context) 