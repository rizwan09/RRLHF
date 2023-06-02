from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MAX_T5_TOKENS = 500

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-xl", cache_dir="/fs/scratch/rng_cr_rtc_hmi_gpu_user_c_lf/pmd1syv/chatgpt-qa-car-manuals/resources/pretrained_models", local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-xl", cache_dir="/fs/scratch/rng_cr_rtc_hmi_gpu_user_c_lf/pmd1syv/chatgpt-qa-car-manuals/resources/pretrained_models", local_files_only=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def CutTokens(content: str, maxTokens: int):
    tokens = content.split(" ")
    if len(tokens) > maxTokens:
        del tokens[maxTokens:]
    return " ".join(tokens)


question = "who wrote the treasure of the sierra madre"
context="The Treasure of the Sierra Madre (film)\"\nThe Treasure of the Sierra Madre (film) The Treasure of the Sierra Madre is a 1948 American dramatic adventurous neo-western written and directed by John Huston. It is an adaptation of B. Traven's 1927 novel of the same name, set in the 1920s, in which, driven by their desperate economic plight, two young men, Fred C. Dobbs (Humphrey Bogart) and Bob Curtin (Tim Holt), join old-timer Howard (Walter Huston, the director's father) in Mexico to prospect for gold. \"\"The Treasure of the Sierra Madre\"\" was one of the first Hollywood productions to be shot on location outside the United States"
context = CutTokens(context, MAX_T5_TOKENS - question.count(" ") - 1)
prompt = "Context: " + context + '\n\n ' + "Question: " + question

prompt2feed_tokenized = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**prompt2feed_tokenized, max_new_tokens=MAX_T5_TOKENS)
answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)

import pdb
pdb.set_trace()






