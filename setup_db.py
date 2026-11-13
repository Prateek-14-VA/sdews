from app import create_app, db
from app.models import User

app = create_app('development')

with app.app_context():
    # Delete the existing admin user if it's there
    existing_user = User.query.filter_by(username='admin').first()
    if existing_user:
        db.session.delete(existing_user)
        db.session.commit()
        print("Deleted old admin user.")
    
    # Create a fresh admin user
    u = User(username='admin', role='hod')
    u.set_password('admin123')
    db.session.add(u)
    db.session.commit()
    print("✅ Fresh admin user created with password: admin123")