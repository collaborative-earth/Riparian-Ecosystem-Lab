from pathlib import Path

import click
import yaml

import dam_model_funcs
from DamDataGenerator import DataGenerator
from perf_analysis_funcs import plot_training, do_perf_analysis
from train import train_model


@click.group()
def cli():
    pass


@cli.command()
@click.option("--hyperparameters", required=True, help="path to yml file that contains hyperparameter config.")
@click.option("--input-data", required=True, help="Path to input data directory (with expected npy and pkl files).")
@click.option("--artifact-dir", required=False, help="Root path to store all training artifacts.")
def train(
        hyperparameters: str,
        input_data: str,
        artifact_dir: str
):
    click.echo('Training model...')
    with open(hyperparameters) as h:
        options = yaml.load(h, yaml.Loader)

    data_params = options["data_params"]
    data_generator = DataGenerator(data_path=Path(input_data), **data_params)

    model = dam_model_funcs.build_model(options)

    training_params = options["training_params"]
    trained_model, model_output_path = train_model(
        model,
        data_generator,
        options["model_type"],
        training_params,
        output_path=Path(artifact_dir)
    )
    plot_training(model_output_path)
    do_perf_analysis(trained_model, model_output_path)


@cli.command()
@click.option("--model-path", required=True, help="Path to trained model file.")
def infer():
    click.echo('Inferring with model...')


if __name__ == '__main__':
    cli()
