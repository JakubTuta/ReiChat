import os

# Set before anything imports app.db — the engine is built from these at import
# time. Forced rather than defaulted: a test run must never touch the database
# the app itself uses, whatever DB_NAME is set to in the shell.
os.environ["DB_NAME"] = "chatbot_test"
os.environ["DB_NULL_POOL"] = "true"
