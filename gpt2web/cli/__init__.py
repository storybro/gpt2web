import os

from appdirs import user_data_dir

import click
import click_config_file

from gpt2web.cli.config import Config
from gpt2web.models.manager import ModelManager
from gpt2web.models.registry import ModelRegistry

from .commands.models import models
from .commands.start import start


data_dir = user_data_dir("gpt2web", "gpt2web")


@click.group()
@click.option('-r', '--model-registry',
              default="https://raw.githubusercontent.com/storybro/torrents/master/models.json")
@click.option('-m', '--models-path',
              show_default=os.path.join(data_dir, "models"),
              default=os.path.join(data_dir, "models"))
@click.pass_context
@click_config_file.configuration_option()
def cli(ctx, model_registry, models_path):
    Config.model_registry = ModelRegistry(model_registry)
    Config.models_path = models_path
    Config.model_manager = ModelManager(models_path)
    ctx.obj = Config


def ep():
    cli.add_command(start)
    cli.add_command(models)
    cli()
