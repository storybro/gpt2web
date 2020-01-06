from flask import request, jsonify

from gpt2web.generation.generator import GPT2Generator
from .app import app


@app.route("/", methods=['POST'])
def gen():
    generator: GPT2Generator = app.generator
    response = generator.generate(request.get_data().decode('utf-8'))
    return jsonify({'response': response})

