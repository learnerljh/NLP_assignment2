from DataHelper import get_dataset
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import evaluate
import numpy as np
from datasets import load_dataset
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
# Set environment variables for Hugging Face mirror instance
os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
os.environ["WANDB_PROJECT"] = "sequence-classification"  # Set project name for Weights and Biases (WandB)

# Initialize logger
logger = logging.getLogger(__name__)

# Define data-related arguments using dataclasses
@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default='laptop_sup',
        metadata={"help": "The name of the task to train on "},  # Help string for the argument
    )
    max_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."  # Explanation of max sequence length
            )
        },
    )

# Define model-related arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}  # Path to model
    )
    num_labels: int = field(
        metadata={"help": "The number of labels in the dataset"}  # Number of labels in the classification task
    )

# Main function to run the training/evaluation pipeline
def main():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    import torch
    torch.cuda.empty_cache()  # Clear the CUDA cache to free up memory
    
    # Parse command-line arguments into dataclasses
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    set_seed(41)  # Set random seed for reproducibility
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # Parse the arguments
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],  # Log output to the console
    )
    log_level = logging.INFO  # Set log level to INFO
    logger.setLevel(log_level)  # Apply log level to the logger
    datasets.utils.logging.set_verbosity(log_level)  # Set log verbosity for datasets library
    transformers.utils.logging.set_verbosity(log_level)  # Set log verbosity for transformers library
    transformers.utils.logging.enable_default_handler()  # Enable default logging handlers
    transformers.utils.logging.enable_explicit_format()  # Enable explicit format for logging
    
    # Log training and model parameters
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    
    # Load dataset using custom get_dataset function
    dataset = get_dataset(data_args.task_name, ',')  # Load the dataset
    logger.info(f"Dataset: {data_args.task_name}")
    logger.info(f"Dataset size: {len(dataset['train'])}")  # Log the size of the training dataset
    
    # Load pretrained model configuration and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=model_args.num_labels,  # Set the number of labels in the configuration
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,  # Load tokenizer from the model path
    )
    
    # Tokenize the dataset
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding=False, max_length=data_args.max_length),  # Apply tokenization to the dataset
        batched=True,  # Process batches of data at a time
    )
    
    # Determine the device to use (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare train and test datasets
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Load pretrained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,  # Use the loaded configuration
    )
    model.to(device)  # Move the model to the appropriate device (GPU or CPU)
    
    # Create a data collator to handle padding
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)  # Pad sequences to multiples of 8
    
    # Function to compute evaluation metrics
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions  # Extract predictions
        preds = np.argmax(preds, axis=1)  # Get the predicted class labels
        
        # Load evaluation metrics
        accuracy_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')
        
        # Compute accuracy
        accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)['accuracy']
        
        # Compute F1 scores (micro and macro)
        f1_micro = f1_metric.compute(predictions=preds, references=p.label_ids, average='micro')['f1']
        f1_macro = f1_metric.compute(predictions=preds, references=p.label_ids, average='macro')['f1']
        
        # Return a dictionary of computed metrics
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }
    
    # Clear CUDA cache again
    torch.cuda.empty_cache()
    
    # Initialize the Trainer with model, datasets, and training/evaluation configurations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,  # Pass the metric computation function
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    train_result = trainer.train()
    metrics = train_result.metrics  # Get training metrics
    
    # Log number of training samples
    max_train_samples = len(dataset['train'])
    metrics["train_samples"] = min(max_train_samples, len(dataset['train']))
    
    # Save the model and tokenizer
    trainer.save_model()  # Saves the tokenizer too for easy upload
    
    # Log and save training metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # Save the trainer's state
    
    # Evaluate the model on the test set
    logger.info("*** test ***")
    metrics = trainer.evaluate()  # Evaluate the model
    trainer.log_metrics("eval", metrics)  # Log evaluation metrics
    trainer.save_metrics("eval", metrics)  # Save evaluation metrics

# Entry point for the script
if __name__ == "__main__":
    main()  # Call the main function