import sys

import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.finetune(
    sess,
    sys.argv[1],
    multi_gpu=True,
    batch_size=32,
    learning_rate=0.0001,
    model_name=sys.argv[2],
    sample_every=10000,
    max_checkpoints=8,
    save_every=200,
    steps=1000,
)

gpt2.generate(sess)
