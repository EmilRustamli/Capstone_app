from app import app, db

# Initialize database
with app.app_context():
    db.create_all()
    print("Database tables created successfully!") 