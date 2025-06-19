# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch

from tqdm.auto import tqdm


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, tokenizer=tokenizer, seqlen=model.seqlen
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 




# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    progress_bar = tqdm(range(0, nsamples, bs), total=nsamples//bs, desc="Evaluating")
    # Loop through each batch
    for i in progress_bar:
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    #torch.cuda.empty_cache()

    return ppl.item()

# "boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"
def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    
    ########limit = 20
    
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 


















#####################
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
from typing import List, Dict
import logging
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_pad_token(tokenizer, model):
    """
    Ensures that the tokenizer has a pad_token. If not, it assigns the eos_token or adds a new pad_token.

    Parameters:
    - tokenizer: The tokenizer to check and modify.
    - model: The model to resize token embeddings if a new pad_token is added.

    Returns:
    - tokenizer: The updated tokenizer with pad_token set.
    """
    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad_token. Attempting to assign eos_token as pad_token.")
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Assigned eos_token ({tokenizer.pad_token}) as pad_token.")
        else:
            logger.info("Tokenizer does not have an eos_token either. Adding a new pad_token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))  # Important: Resize embeddings
            logger.info(f"Added new pad_token: {tokenizer.pad_token}")
    else:
        logger.info(f"Tokenizer already has a pad_token: {tokenizer.pad_token}")
    return tokenizer

def eval_zero_shot_new(model_name, model, tokenizer, max_samples, task_list = ["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], accelerate = False, batch_size = 16):
    """
    Evaluates a given model on a list of tasks in a zero-shot setting using Hugging Face's Accelerate for optimized performance.

    Parameters:
    - model_name (str): Name of the model.
    - model: Pretrained model for evaluation.
    - tokenizer: Tokenizer corresponding to the model.
    - task_list (List[str]): List of task names to evaluate on.
    - batch_size (int): Batch size for evaluation.

    Returns:
    - results (Dict): Dictionary containing evaluation metrics for each task.
    """
    results = {}
    # Ensure pad_token is set
    tokenizer = ensure_pad_token(tokenizer, model)
    device = model.device
        
    # Initialize the accelerator
    if accelerate:
        accelerator = Accelerator()
        logger.info(f"Using device: {accelerator.device}")
        # Prepare the model with the accelerator
        model, tokenizer = accelerator.prepare(model, tokenizer)
        model.eval()
    else:
        logger.info("Accelerate is not enabled. Using CPU/GPU for evaluation.")
    
   

    for task in task_list:
        logger.info(f"Evaluating on task: {task}")
        
        # Determine max samples for this task
        if isinstance(max_samples, dict):
            task_max = max_samples.get(task_lower, None)
        elif isinstance(max_samples, int):
            task_max = max_samples
        else:
            task_max = None  # No limit
            
        if task.lower() == "boolq":
            dataset = load_dataset("boolq")
            metric = load_metric("accuracy")
            for split in ["validation"]:
                eval_dataset = dataset[split]
                
                if task_max is not None:
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), task_max)))
                    logger.info(f"Evaluating {len(eval_dataset)} samples for task '{task}' split '{split}'.")

                inputs = []
                references = []
                for example in eval_dataset:
                    question = example["question"]
                    passage = example["passage"]
                    input_text = f"Does the following passage answer the question? Passage: {passage} Question: {question}"
                    inputs.append(input_text)
                    references.append(1 if example["answer"] else 0)  # Assuming 'answer' is boolean

                # Batch processing
                dataloader = DataLoader(inputs, batch_size=batch_size)
                predictions = []
                for batch in dataloader:
                    if accelerate:
                        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
                    else:
                        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        outputs = model(**encoding)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        predictions.extend(preds.cpu().numpy())
                        
                print(predictions)
                print(references)
                acc = metric.compute(predictions=predictions, references=references)
                results.setdefault(task, {})[split] = acc

        elif task.lower() == "rte":
            dataset = load_dataset("glue", "rte")
            metric = load_metric("accuracy")
            for split in ["validation"]:
                eval_dataset = dataset[split]
                
                if task_max is not None:
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), task_max)))
                    logger.info(f"Evaluating {len(eval_dataset)} samples for task '{task}' split '{split}'.")
 
 
                inputs = []
                references = []
                for example in eval_dataset:
                    premise = example["sentence1"]
                    hypothesis = example["sentence2"]
                    input_text = f"Premise: {premise} Hypothesis: {hypothesis}"
                    inputs.append(input_text)
                    references.append(example["label"])

                # Batch processing
                dataloader = DataLoader(inputs, batch_size=batch_size)
                predictions = []
                for batch in dataloader:
                    if accelerate:
                        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
                    else:
                        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        outputs = model(**encoding)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        predictions.extend(preds.cpu().numpy())
                acc = metric.compute(predictions=predictions, references=references)
                results.setdefault(task, {})[split] = acc

        elif task.lower() in ["hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"]:
            # These are multiple-choice tasks
            if task.lower() == "hellaswag":
                dataset = load_dataset("hellaswag")
                metric = load_metric("accuracy")
            elif task.lower() == "winogrande":
                dataset = load_dataset("winogrande", "winogrande_xl")
                metric = load_metric("accuracy")
            elif task.lower() in ["arc_challenge", "arc_easy"]:
                subset = "ARC-Challenge" if task.lower() == "arc_challenge" else "ARC-Easy"
                dataset = load_dataset("arc", subset)
                metric = load_metric("accuracy")
            elif task.lower() == "openbookqa":
                dataset = load_dataset("openbookqa", "main")
                metric = load_metric("accuracy")
            else:
                logger.warning(f"Task {task} not recognized for multiple-choice evaluation.")
                continue

            for split in ["validation", "test"]:
                if split not in dataset:
                    logger.info(f"Split '{split}' not found for task '{task}'. Skipping.")
                    continue
                eval_dataset = dataset[split]
                
                if task_max is not None:
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), task_max)))
                    logger.info(f"Evaluating {len(eval_dataset)} samples for task '{task}' split '{split}'.")
 
                inputs = []
                references = []
                option_texts = []
                for example in eval_dataset:
                    if task.lower() == "hellaswag":
                        context = example["ctx"]
                        ending_choices = example["endings"]
                        label = example["label"]
                        # Format: "Context: {context} Choice: {choice}"
                        formatted_choices = [f"{context} {choice}" for choice in ending_choices]
                    elif task.lower() == "winogrande":
                        sentence = example["sentence"]
                        choices = example["options"]
                        label = example["answer"]
                        formatted_choices = [f"{sentence} {choice}" for choice in choices]
                    elif task.lower() in ["arc_challenge", "arc_easy"]:
                        question = example["question"]
                        choices = example["choices"]["text"]
                        label = example["answerKey"]
                        formatted_choices = [f"{question} {choice}" for choice in choices]
                    elif task.lower() == "openbookqa":
                        question = example["question"]
                        choices = example["choices"]
                        label = example["answerKey"]
                        formatted_choices = [f"{question} {choice['text']}" for choice in choices]
                    else:
                        continue
                    inputs.append(formatted_choices)
                    # Convert label to index
                    if task.lower() in ["hellaswag", "winogrande"]:
                        references.append(label)
                    else:
                        references.append(ord(label.upper()) - ord('A'))  # Convert 'A'-'D' to 0-3

                # Flatten inputs for batch processing
                flat_inputs = [choice for sublist in inputs for choice in sublist]
                dataloader = DataLoader(flat_inputs, batch_size=batch_size)
                scores = []
                for batch in dataloader:
                    if accelerate:
                        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
                    else:
                        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        outputs = model(**encoding)
                        logits = outputs.logits
                        # Assuming the model outputs logits for each class; adjust if necessary
                        # For multiple-choice, models often output logits per choice
                        scores.extend(logits.cpu().numpy())

                # Reshape scores to (num_examples, num_choices)
                num_choices = len(inputs[0]) if inputs else 0
                scores = torch.tensor(scores).view(-1, num_choices)
                predictions = torch.argmax(scores, dim=1).cpu().numpy()

                acc = metric.compute(predictions=predictions, references=references)
                results.setdefault(task, {})[split] = acc

        else:
            logger.warning(f"Task {task} is not supported.")
            continue

    # Print the results
    for task, splits in results.items():
        print(f"Task: {task}")
        for split, metric_result in splits.items():
            print(f"  {split}: {metric_result}")
    return results