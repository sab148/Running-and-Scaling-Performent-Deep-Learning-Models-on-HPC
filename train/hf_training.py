import argparse
import sys

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def get_model(name: str = "gpt2"):
    # In HF, it's easy to load models for different downstream tasks:
    #
    # https://huggingface.co/docs/transformers/model_doc/auto
    #
    #
    # For example, here we do text classification. More examples for NLP:
    #
    # https://huggingface.co/docs/transformers/v5.0.0rc1/en/model_doc/auto#natural-language-processing
    #
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(name)

    # GPT-style models have no pad token by default.
    # For batching, we reuse EOS as PAD (common HF workaround)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def get_data(tokenizer,
             name: str = "imdb",
             nsample: int = 1000,
             seed: int = 42):
    # imdb dataset: sentiment analysis of movie reviews: text -> {0,1}
    #
    # https://huggingface.co/datasets/stanfordnlp/imdb
    #
    # Note: We intentionally subsample the dataset to keep the example
    # fast and suitable for a tutorial

    dataset = load_dataset(name)
    d_train = dataset["train"]\
        .shuffle(seed=seed)\
        .select(range(nsample))
    d_val = dataset["test"]\
        .shuffle(seed=seed)\
        .select(range(nsample))

    d_train = d_train.map(
        lambda x: tokenizer(
            x['text'],
            truncation=True,
        ),
        batched=True)
    d_val = d_val.map(
        lambda x: tokenizer(
            x['text'],
            truncation=True,
        ),
        batched=True)

    return d_train, d_val


def _opts():
    p = argparse.ArgumentParser(
        prog = sys.argv[0],
        description = "Basic HF example")

    p.add_argument(
        '--training_type', '-t',
        default = 'ddp',
        choices = ('ddp', 'fsdp'),
        help = 'type of training'
    )

    return p


if '__main__' == __name__:
    args = _opts().parse_args()
    set_seed(42)

    model, tokenizer = get_model()
    d_train, d_val = get_data(tokenizer)

    if 'ddp' == args.training_type:
        # DDP parallelization (when on several nodes)
        training_args = TrainingArguments(
            output_dir = "./hf_example",
            overwrite_output_dir = True,
            save_strategy = 'no',
            # For tutorial/debugging purposes, we limit training by steps
            # `max_steps` overrides `num_train_epochs`
            max_steps = 10,
            num_train_epochs = 0,
            learning_rate = 2e-5,
            per_device_train_batch_size = 4,
            bf16 = True,
            accelerator_config = dict(
                dispatch_batches = False,
            ),
        )
    elif 'fsdp' == args.training_type:
        # FSDP parallelization
        training_args = TrainingArguments(
            output_dir = "./hf_example",
            overwrite_output_dir = True,
            save_strategy = 'no',
            num_train_epochs = 3,
            max_steps = 10,
            learning_rate = 2e-5,
            per_device_train_batch_size = 4,
            bf16 = True,
            accelerator_config = dict(
                dispatch_batches = False,
            ),
            fsdp = "full_shard auto_wrap",
            fsdp_config = dict(
                fsdp_activation_checkpointing = True
            ),
            gradient_checkpointing = False,
            evaluation_strategy="steps",
            eval_steps=5,
            logging_steps=1,
        )
    else:
        raise ValueError(f"unknown --training_type={args.training_type}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=d_train,
        eval_dataset=d_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    ## model inference
    #
    # model = AutoModelForSequenceClassification.from_pretrained(<model_dir>)
    # tokenizer = ...
    #
    # reviews = tokenizer(reviews, ...)
    # logits = model(**reviews).logits
    # probs = torch.softmax(logits, dim=-1)
    # pred = torch.argmax(probs, dim=-1)

    # for more examples, see https://gitlab.jsc.fz-juelich.de/sdlaml/hf_llm
    #
    # Note: Some parallelization strategies are implemented only for
    # forward-passes only (i.e. for inference)
