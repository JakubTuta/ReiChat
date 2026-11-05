from app.core.logging import configure_logging
from app.main import create_app

# Kept out of main.py so importing create_app (tests do) never reconfigures
# logging — that would detach pytest's log capture.
configure_logging()
app = create_app()
