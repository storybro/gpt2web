import os

import click
from pyngrok import ngrok

from gpt2web.api.app import app
from gpt2web.generation.generator import GPT2Generator


@click.command()
@click.argument('model-name')
@click.option('--force-cpu', '-f', is_flag=True, help="Force the model to run on the CPU")
@click.pass_obj
def start(config, model_name, force_cpu):

    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = config.model_manager.models.get(model_name)
    if not model:
        click.echo(f"Model `{model_name}` is not installed.")
        return

    app.generator = GPT2Generator(model)

    # Open a tunnel on the default port 80
    public_url = ngrok.connect(port=5000)

    print(public_url)

    app.run()
