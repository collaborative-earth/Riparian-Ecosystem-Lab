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
@click.option("--finetune-checkpoint", required=False, help="path of pre-trained model to fine-tune.")
@click.option("--input-data", required=True, help="Path to input data directory (with expected npy and pkl files).")
@click.option("--artifact-dir", required=True, help="Root path to store all training artifacts.")
@click.option("--dev", is_flag=True, help="Flag to use two batches of data during training.")
def train(
        hyperparameters: str,
        finetune_checkpoint: str,
        input_data: str,
        artifact_dir: str,
        dev: bool
):
    click.echo('Training model...')
    with open(hyperparameters) as h:
        options = yaml.load(h, yaml.Loader)

    data_params = options["data_params"]
    data_generator = DataGenerator(data_path=Path(input_data), dev=dev, **data_params)

    if finetune_checkpoint:
        model = dam_model_funcs.load_pretrained_model(finetune_checkpoint)
    else:
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

    test_data_generator = DataGenerator(data_path=Path(input_data), batch_size=128, test=True, RGBN=[0, 1, 2, 3])
    do_perf_analysis(trained_model, test_data_generator, model_output_path)


@cli.command()
@click.option("--model-path", required=True, help="Path to trained model file.")
def infer():
    click.echo('Inferring with model...')


if __name__ == '__main__':
    cli()
