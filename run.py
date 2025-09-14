import os
import sys
from app import create_app, db
from models.database import init_sample_data
from flask_migrate import Migrate

app = create_app()
migrate = Migrate(app, db)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--init-db':
        with app.app_context():
            print("Creating database tables...")
            db.create_all()
            print("Initializing sample data...")
            init_sample_data()
            print("Database initialization complete!")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)