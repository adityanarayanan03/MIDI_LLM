'''
Main run code. Some development done in jupyter notebooks to make things 
easier for now with loading datasets, but it should get populated here for 
the final product.
'''

from midi_llm import LLM, Llama_Tokenizer, MIDI_Dataset

from tqdm import tqdm

#logging setup
import logging
logging.basicConfig()
logger = logging.getLogger("MAIN")
logger.setLevel(logging.DEBUG)

#Setup parameters
CONTEXT_LEN = 100

#instantiate our wrappers around various dataset and huggingface functions
midi_dataset = MIDI_Dataset("MIDI_LLM/OriginalMidiFiles/bach/cellosui", tokenization_method='TSD')
llama = LLM("config.yaml")
llama_tok = Llama_Tokenizer("config.yaml")

for track in tqdm(midi_dataset, desc="Tracks", position=0):
    if len(track) <= 1025:
        continue

    '''
    Track is now a set of tokens in the llama-token space. We want to convert
    this to pseudo-english since our huggingface inference endpoint does not 
    support passing tokens directly. Essentially, we're detokenizing since 
    they'll tokenize anyway.
    '''
    correct = 0
    total = 0

    inner_bar = tqdm(range(0, len(track) - 1025), desc=f"Single Track", position=1, leave=True)
    for idx in inner_bar:

        #attempting to use largest context hugging face endpoint will allow
        context = track[idx: idx + CONTEXT_LEN]
        true_next_token = track[idx+CONTEXT_LEN]
        #logger.debug(f"Context from track: {context}, length of context is {len(context)}")

        pseudo_english = llama_tok.detokenize(context)
        #logger.debug(f"Generated pseudo_english prompt: {pseudo_english}")

        out, tokens = llama.infer(pseudo_english)
        token_out = tokens[0]

        if token_out.id == true_next_token:
            #logger.debug(f"Llama returned next tokens: {tokens}, true next token is {true_next_token}")
            correct += 1
        total += 1
        #logger.info(f"Current Accuracy: {correct / total}")
        inner_bar.set_postfix_str(f"Acc: {correct/total:.4f}")