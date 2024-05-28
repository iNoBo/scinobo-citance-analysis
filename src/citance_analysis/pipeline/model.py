import os
import torch
import importlib
from tqdm import tqdm
from peft import PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

BASE_PATH = importlib.resources.files(__package__.split(".")[0])

batch_size = int(os.getenv('BSIZE', 16))

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", cache_dir=os.path.join(BASE_PATH, "models_cache"), torch_dtype="auto", device_map="auto")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", cache_dir=os.path.join(BASE_PATH, "models_cache"))

# HERE WE CAN ADD A LORA MODEL (e.g. ARTIFACT EXTRACTION)
if torch.cuda.is_available():
    model = PeftModel.from_pretrained(model, os.path.join(BASE_PATH, "model_checkpoints/pilot/citance_analysis_lora_flan_t5"), device_map={'': 0})
else:
    model = PeftModel.from_pretrained(model, os.path.join(BASE_PATH, "model_checkpoints/pilot/citance_analysis_lora_flan_t5"), device_map={'': 'cpu'})

print(f"Running merge_and_unload")
model = model.merge_and_unload()

print(model.hf_device_map)
def infer_text_web(sentence, max_len, **kwargs):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
    
    if max_len == 0:
        outputs = model.generate(input_ids=inputs, **kwargs)
    else:
        outputs = model.generate(input_ids=inputs, max_new_tokens=max_len, **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def infer_text(sentence, **kwargs):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, **kwargs)
    if 'return_dict_in_generate' in kwargs:
        return outputs
    return [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]


def tokenize_text(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)


def compute_relative_probs(choices, predictions):
    abs_probs = {}
    for choice in choices:
        choice_score = 0.0
        tokenized_choice = [tok[1:] if tok.startswith('▁') else tok for tok in tokenizer._tokenize(choice)]
        for tok_i, tok in enumerate(tokenized_choice):
            if tok_i == len(predictions):
                break
            if tok in predictions[tok_i]:
                choice_score += predictions[tok_i][tok]
        abs_probs[choice] = choice_score
    
    Z = sum(abs_probs.values())
    if Z < 0.6:
        print(f"{1-Z} of unaccounted probability in classify")
        print(str(predictions))
        print(str(abs_probs))

    rel_probs = (
        {choice: prob / Z for (choice, prob) in abs_probs.items()}
        if Z != 0.0
        else abs_probs
    )
    return rel_probs


def classify_text(prompt, choices, top_k, **kwargs):
    if 'output_scores' not in kwargs:
        kwargs['output_scores'] = True
    if 'return_dict_in_generate' not in kwargs:
        kwargs['return_dict_in_generate'] = True
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, **kwargs)

    predictions = [[[tokenizer.decode(x[1].item()), torch.exp(x[0]).item()] for x in zip(outputs.scores[i][0].topk(top_k).values, outputs.scores[i][0].topk(top_k).indices)] for i in range(len(outputs.scores))]
    # Solve the bug where some indices have the same token value (e.g. Yes has 2 indices)
    predictions_dict = []
    for prediction in predictions:
        prediction_dict = dict()
        for token, prob in prediction:
            if token in prediction_dict:
                prediction_dict[token] += prob
            else:
                prediction_dict[token] = prob
        predictions_dict.append(prediction_dict)

    rel_probs = compute_relative_probs(choices, predictions_dict)

    return rel_probs


def _classify_text_batch(prompts, choices, top_k, **kwargs):
    if 'output_scores' not in kwargs:
        kwargs['output_scores'] = True
    if 'return_dict_in_generate' not in kwargs:
        kwargs['return_dict_in_generate'] = True

    # Batch processing of prompts
    inputs = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(input_ids=inputs['input_ids'], **kwargs)

    all_rel_probs = []
    for p_i, prompt in enumerate(prompts):
        predictions = [[[tokenizer.decode(x[1].item()), torch.exp(x[0]).item()] for x in zip(outputs.scores[i][p_i].topk(top_k).values, outputs.scores[i][p_i].topk(top_k).indices)] for i in range(len(outputs.scores))]
        # Solve the bug where some indices have the same token value (e.g. Yes has 2 indices)
        predictions_dict = []
        for prediction in predictions:
            prediction_dict = dict()
            for token, prob in prediction:
                if token in prediction_dict:
                    prediction_dict[token] += prob
                else:
                    prediction_dict[token] = prob
            predictions_dict.append(prediction_dict)

        rel_probs = compute_relative_probs(choices, predictions_dict)

        all_rel_probs.append(rel_probs)

    return all_rel_probs


def classify_text_batch(prompts, choices, top_k, batch_size=batch_size, **kwargs):
    # Initialize the list to store all results
    all_results = []
    
    # Calculate the number of batches
    num_batches = len(prompts) // batch_size + (0 if len(prompts) % batch_size == 0 else 1)
    
    # Process in batches with progress bar
    for i in tqdm(range(num_batches), desc="Processing batches"):
        # Calculate start and end index of the current batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Slice the batch
        batch_prompts = prompts[start_idx:end_idx]
        
        # Call the classification function for the current batch
        batch_results = _classify_text_batch(batch_prompts, choices, top_k, **kwargs)
        
        # Extend the overall results with the current batch's results
        all_results.extend(batch_results)
    
    return all_results


def multilabel_classify_text(prompt, choices, top_k, **kwargs):
    # Get the probabilities for each choice using a Yes/No classifier from "classify_text" function
    choice_probs = dict()
    for choice in choices:
        modded_prompt = f"{prompt}\nAnswer: {choice}\nQuestion: Is the answer given above correct, Yes or No?"
        classify_res = classify_text(modded_prompt, ['Yes', 'No'], top_k, **kwargs)
        choice_probs[choice] = classify_res['Yes']
    return choice_probs


if __name__ == '__main__':
    res = infer_text("### Snipet: In contrast to the recommendations given by Johnson [15], our study found that the use of their data cleaning method led to a loss of critical information. ### Citation Mark: [15] ### Question: What is the polarity of the citation? ### Answer:")
    print()
