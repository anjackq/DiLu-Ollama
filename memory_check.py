import yaml
from rich import print
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.runtime import configure_runtime_env

# 1. Load the configuration
try:
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
except FileNotFoundError:
    print("[red]Error: config.yaml not found. Make sure you are in the DiLu root directory.[/red]")
    exit(1)

# 2. Setup Environment Variables (Critical Step)
selected_model = configure_runtime_env(config)
if config['OPENAI_API_TYPE'] == 'ollama':
    print(f"[yellow]Configured for Local Ollama: {selected_model}[/yellow]")
elif config['OPENAI_API_TYPE'] == 'gemini':
    print(f"[yellow]Configured for Gemini API: {selected_model}[/yellow]")

# 3. Initialize and Check Memory
try:
    print(f"[cyan]Loading memory from: {config['memory_path']}...[/cyan]")
    memory = DrivingMemory(db_path=config['memory_path'])

    # Check total items
    count = memory.scenario_memory._collection.count()
    print(f"[green]Total memories found: {count}[/green]")

    # Peek at the data (first item)
    if count > 0:
        print("\n[bold]Sample Memory Item:[/bold]")
        peek_data = memory.scenario_memory._collection.peek()
        print(peek_data)
    else:
        print("[yellow]The memory database is empty.[/yellow]")

except Exception as e:
    print(f"[red]Error loading memory:[/red] {e}")
    print("[yellow]Hint: Embedding dimension mismatches are common when switching models/providers. Use a separate memory folder per embedding model (for example, `memories/qwen3_embed_8b`).[/yellow]")
