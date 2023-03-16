import click
import yaml

import dam_model_funcs
from DamDataGenerator import DataGenerator
from train import train


@click.group()
def cli():
    pass


@cli.command()
@click.option("--options", required=True, help="path to yml file that contains hyperparameter config.")
@click.option("--input-data", required=True, help="Path to input data directory (with expected npy and pkl files).")
@click.option("--model-id", required=True, help="Id of model to use in artifact file paths / names.")
@click.option("--artifact-dir", required=False, help="Root path to store all training artifacts.")
def run_training(
        options_path: str,
        input_data_path: str,
        model_id: str,
        artifact_dir: str
):
    click.echo('Training model...')
    with open(options_path) as h:
        options = yaml.load(h)

    data_params = options["data_params"]
    data_generator = DataGenerator(**data_params)

    model = dam_model_funcs.build_model(**options)

    training_params = options["training_params"]
    trained_model = train(model, data_generator, options["model_type"], training_params)




@cli.command()
@click.option("--model-path", required=True, help="Path to trained model file.")
def infer():
    click.echo('Inferring with model...')


if __name__ == '__main__':
    cli()
