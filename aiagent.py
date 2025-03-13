import asyncio
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Define Ollama Model for Pydantic AI
ollama_model = OpenAIModel(
    model_name='llama3:latest',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')  # Ensure Ollama API is running
)

def load_csv(file_path):
    """Loads CSV file into a Pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

async def search_csv(query, data):
    """Queries the dataset using Ollama's local LLM via pydantic_ai."""
    if data is None:
        return "Error: No data available."

    system_prompt = f"""You are an expert in analyzing tabular data.
    Below is a dataset from a CSV file:
    
    {data.to_string(index=False)}
    
    Answer concisely based on this data. The user's question is: {query}
    Provide only relevant insights without adding unnecessary information."""

    agent = Agent(
        ollama_model,
        system_prompt=system_prompt,
    )

    response = await agent.run(user_prompt=query)
    return response.data

def query_csv(user_query):
    """Handles user queries via Ollama."""
    file_path = "/Users/nawdeepkumar/gradio_csv_app/Housing.csv"  # Update this path
    data = load_csv(file_path)

    if data is None:
        return "Error: CSV file not found or couldn't be loaded."

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(search_csv(user_query, data))
    
    return response

def plot_graph(x_col, y_col):
    """Plots a graph from the CSV file."""
    file_path = "/Users/nawdeepkumar/gradio_csv_app/Housing.csv"  # Update this path
    data = load_csv(file_path)

    if data is None:
        return "Error: CSV file not found or couldn't be loaded.", None

    if x_col not in data.columns or y_col not in data.columns:
        return f"Error: Columns '{x_col}' or '{y_col}' not found in the CSV.", None

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x=x_col, y=y_col, color='blue', alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")

    graph_path = "/Users/nawdeepkumar/gradio_csv_app/plot.png"
    plt.savefig(graph_path)
    plt.close()

    return f"Graph for {x_col} vs {y_col}", graph_path

def create_gradio_ui():
    with gr.Blocks() as ui:
        gr.Markdown("# 📊 CSV Query & Visualization Tool")
        
        with gr.Tab("Query CSV"):
            user_query = gr.Textbox(label="Enter Query", placeholder="Example: What is the average price?")
            query_button = gr.Button("Ask Ollama")
            query_output = gr.Textbox(label="Response")
            query_button.click(query_csv, inputs=user_query, outputs=query_output)

        with gr.Tab("Graph Visualization"):
            col1 = gr.Dropdown(label="Select X-axis Column", choices=["price", "area", "bedrooms"], interactive=True)
            col2 = gr.Dropdown(label="Select Y-axis Column", choices=["price", "area", "bedrooms"], interactive=True)
            plot_button = gr.Button("Generate Graph")
            plot_output = gr.Image(label="Generated Graph")
            plot_button.click(plot_graph, inputs=[col1, col2], outputs=[gr.Textbox(), plot_output])

    return ui

if __name__ == "__main__":
    ui = create_gradio_ui()
    ui.launch()
