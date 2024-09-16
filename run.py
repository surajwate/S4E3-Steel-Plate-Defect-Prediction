import subprocess
from src import model_dispatcher

# Get the models from model_dispatcher
models = model_dispatcher.models.keys()

# Run each model from model_dispatcher
for model in models:
    print(f"Running model: {model}")
    subprocess.run(["python", "main.py", "--model", model])
