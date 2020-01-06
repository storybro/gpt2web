from flask import Flask

from gpt2web.generation.generator import GPT2Generator

app = Flask('gpt2web')
app.generator: GPT2Generator = None
