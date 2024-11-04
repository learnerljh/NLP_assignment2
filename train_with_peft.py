import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaPreTrainedModel, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
from DataHelper import get_dataset
import logging
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
os.environ["WANDB_PROJECT"]="sequence-classification-peft"

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.down_projection = nn.Linear(input_dim, hidden_dim)
        self.up_projection = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        down_projected = self.down_projection(x)
        activated = self.activation(down_projected)
        up_projected = self.up_projection(activated)
        return up_projected

class RobertaWithAdapter(RobertaPreTrainedModel):
    def __init__(self, config, adapter_hidden_dim=256):
        super(RobertaWithAdapter, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.adapters = nn.ModuleList([Adapter(config.hidden_size, adapter_hidden_dim) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 冻结 RoBERTa 模型的参数
        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        adapted_hidden_states = []
        for i, layer_hidden_state in enumerate(hidden_states[1:]):
            adapted_hidden_state = self.adapters[i](layer_hidden_state)
            adapted_hidden_states.append(layer_hidden_state + adapted_hidden_state)

        # 使用最后一层的适应后隐藏状态
        pooled_output = adapted_hidden_states[-1][:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    num_labels: int = field(
        metadata={"help": "The number of labels in the dataset"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_name: Optional[str] = field(
        default='laptop_sup',
        metadata={"help": "The name of the task to train on "},
    )
    max_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    hidden_dim: int = field(
        default=256,
        metadata={"help": "The hidden dimension of the adapter"}
    )




def main():
    import torch
    torch.cuda.empty_cache()
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    set_seed(42)
    data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    dataset = get_dataset(data_args.task_name, ',')
    logger.info(f"Dataset: {data_args.task_name}")
    logger.info(f"Dataset size: {len(dataset['train'])}")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # Load pretrained model and tokenizer
    config = RobertaConfig.from_pretrained('roberta-base', num_labels=data_args.num_labels)
    model = RobertaWithAdapter(config, adapter_hidden_dim=data_args.hidden_dim)
    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base',
    )
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding=False),
        batched=True,
    )
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    model.to(device)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        accuracy_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')
    
        accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)['accuracy']
    
        f1_micro = f1_metric.compute(predictions=preds, references=p.label_ids, average='micro')['f1']
        f1_macro = f1_metric.compute(predictions=preds, references=p.label_ids, average='macro')['f1']
    
        return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
        }

    torch.cuda.empty_cache()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    max_train_samples = len(dataset['train'])

    metrics["train_samples"] = min(max_train_samples, len(dataset['train']))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** test ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
