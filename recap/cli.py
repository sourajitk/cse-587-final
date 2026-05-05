"""
RECAP Command Line Interface.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
import json

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="recap")
def main():
    """RECAP: Reaction Completion for Biochemical Pathways."""
    pass


@main.command()
@click.option(
    "--substrates", "-s",
    help="Substrate SMILES (dot-separated for multiple)",
)
@click.option(
    "--products", "-p",
    help="Product SMILES (dot-separated for multiple)",
)
@click.option(
    "--partial",
    help="Partial reaction string (use ? for missing)",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["forward", "retro", "fill"]),
    default="forward",
    help="Completion mode",
)
@click.option(
    "--ec", "-e",
    help="EC number for enzyme context",
)
@click.option(
    "--model",
    default="recap-base",
    help="Model to use",
)
@click.option(
    "--num-predictions", "-n",
    default=5,
    type=int,
    help="Number of predictions to return",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--use-kg",
    type=click.Path(exists=True),
    help="Path to TSV graph file to use for inference",
)
def complete(
    substrates: Optional[str],
    products: Optional[str],
    partial: Optional[str],
    mode: str,
    ec: Optional[str],
    model: str,
    num_predictions: int,
    output: str,
    use_kg: Optional[str],
):
    """Complete a biochemical reaction."""
    from recap import ReactionCompleter
    
    with console.status("[bold green]Loading model..."):
        completer = ReactionCompleter.from_pretrained(model, mode=mode, kg_path=use_kg)
    
    # Parse inputs
    substrates_list = substrates.split(".") if substrates else None
    products_list = products.split(".") if products else None
    
    # Run completion
    with console.status("[bold green]Generating predictions..."):
        result = completer.complete(
            substrates=substrates_list,
            products=products_list,
            partial_reaction=partial,
            mode=mode,
            ec_number=ec,
            num_predictions=num_predictions,
        )
    
    # Output results
    if output == "json":
        output_dict = {
            "input": result.input_reaction,
            "mode": result.mode,
            "predictions": [
                {"smiles": pred, "score": score}
                for pred, score in zip(result.predictions, result.scores)
            ],
            "confidence": result.confidence,
            "is_valid": result.is_valid,
        }
        console.print_json(json.dumps(output_dict))
    else:
        # Rich text output
        console.print(Panel(
            f"[bold]Input:[/bold] {result.input_reaction}\n"
            f"[bold]Mode:[/bold] {result.mode}",
            title="RECAP Completion",
        ))
        
        table = Table(title="Predictions")
        table.add_column("Rank", style="cyan")
        table.add_column("SMILES", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Valid", style="magenta")
        
        for i, (pred, score) in enumerate(zip(result.predictions, result.scores)):
            is_valid = "✓" if result.is_valid else "✗"
            table.add_row(str(i + 1), pred, f"{score:.4f}", is_valid)
        
        console.print(table)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--mode", "-m",
    type=click.Choice(["forward", "retro", "fill"]),
    default="forward",
)
@click.option(
    "--model",
    default="recap-base",
)
@click.option(
    "--batch-size", "-b",
    default=32,
    type=int,
)
def batch(
    input_file: str,
    output_file: str,
    mode: str,
    model: str,
    batch_size: int,
):
    """Process reactions in batch."""
    import pandas as pd
    from recap import ReactionCompleter
    from tqdm import tqdm
    
    console.print(f"[bold]Loading model: {model}[/bold]")
    completer = ReactionCompleter.from_pretrained(model, mode=mode)
    
    console.print(f"[bold]Loading data: {input_file}[/bold]")
    df = pd.read_csv(input_file)
    
    results = []
    
    with console.status("[bold green]Processing...") as status:
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i + batch_size]
            
            for _, row in batch_df.iterrows():
                if mode == "forward":
                    substrates = row.get("substrates", "").split(".")
                    result = completer.complete(substrates=substrates, mode=mode)
                elif mode == "retro":
                    products = row.get("products", "").split(".")
                    result = completer.complete(products=products, mode=mode)
                else:
                    partial = row.get("reaction", "")
                    result = completer.complete(partial_reaction=partial, mode=mode)
                
                results.append({
                    "input": result.input_reaction,
                    "prediction": result.top_prediction,
                    "confidence": result.confidence,
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    console.print(f"[bold green]Results saved to: {output_file}[/bold]")


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Training configuration file",
)
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True),
    required=True,
    help="Data directory",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    required=True,
    help="Output directory for checkpoints",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    help="Resume from checkpoint",
)
def train(
    config: str,
    data_dir: str,
    output_dir: str,
    resume: Optional[str],
):
    """Train a RECAP model."""
    import yaml
    from pathlib import Path
    
    console.print("[bold]Starting training...[/bold]")
    
    # Load config
    with open(config) as f:
        train_config = yaml.safe_load(f)
    
    console.print(f"Config: {config}")
    console.print(f"Data: {data_dir}")
    console.print(f"Output: {output_dir}")
    
    # Import training module
    try:
        from recap.training import Trainer
        
        trainer = Trainer(
            config=train_config,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        
        if resume:
            trainer.load_checkpoint(resume)
        
        trainer.train()
        
    except ImportError:
        console.print("[yellow]Training module not available. Use scripts/train.py instead.[/yellow]")


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.argument("test_data", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file for results",
)
def evaluate(
    checkpoint: str,
    test_data: str,
    output: Optional[str],
):
    """Evaluate a RECAP model."""
    console.print(f"[bold]Evaluating: {checkpoint}[/bold]")
    console.print(f"Test data: {test_data}")
    
    # Import evaluation module
    try:
        from recap.evaluation import evaluate_model
        
        results = evaluate_model(checkpoint, test_data)
        
        # Display results
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in results.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[bold green]Results saved to: {output}[/bold]")
        
    except ImportError:
        console.print("[yellow]Evaluation module not available. Use scripts/evaluate.py instead.[/yellow]")


@main.command()
def info():
    """Show model and system information."""
    import torch
    import platform
    
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Python Version", platform.python_version())
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU", torch.cuda.get_device_name(0))
    
    console.print(table)
    
    # Model info
    model_table = Table(title="Available Models")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Description", style="green")
    
    model_table.add_row("recap-base", "Base model (250M params)")
    model_table.add_row("recap-large", "Large model (780M params)")
    model_table.add_row("recap-fast", "Fast model (60M params)")
    
    console.print(model_table)


if __name__ == "__main__":
    main()
