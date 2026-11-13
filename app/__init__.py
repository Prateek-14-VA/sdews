"""
Application factory fpor the Student Dropout Early Warning System (SDEWS).

Creates the Flask app, initializes extensions, registers blueprints, and ensures required runtime folders exist.
"""
from pathlib import Path
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate

from .config import config

#----Extensions (created once, initialized in factory)----
db=SQLAlchemy()
login_manager=LoginManager()
migrate=Migrate()

def create_app(config_name: str="default") -> Flask:
    """Create and configure the Flask application."""

    app=Flask(
        __name__,
        instance_relative_config=True,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config[config_name])

    #----Ensure required folders exist at startup ----
    Path(app.instance_path).mkdir(parents=True,exist_ok=True)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True,exist_ok=True)
    Path(app.config["LOG_DIR"]).mkdir(parents=True,exist_ok=True)

     # ---- Init extensions ----
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    login_manager.login_view = "main.login"
    login_manager.login_message_category = "warning"
        # Placeholder user_loader (real implementation in Week 8)
    @login_manager.user_loader
    def load_user(user_id):
        from .models import User
        return User.query.get(int(user_id))

    # ---- Register blueprints ----
    from .routes import main_bp
    app.register_blueprint(main_bp)

    # ---- Health-check endpoint ----
    @app.get("/health")
    def health():
        return {"status": "ok", "app": "SDEWS", "version": "0.1.0"}

    return app