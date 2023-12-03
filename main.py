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
CONTEXT_LEN = 200

tokenization_methods = ['Structured', 'TSD', 'MIDILike']

#instantiate our wrappers around various dataset and huggingface functions
llama = LLM("MIDI_LLM/config.yaml")
llama_tok = Llama_Tokenizer("MIDI_LLM/config.yaml")

for tokenization_method in tokenization_methods:

    midi_dataset = MIDI_Dataset("MIDI_LLM/OriginalMidiFiles/bach/cellosui/", tokenization_method=tokenization_method)

    for index, track in tqdm(enumerate(midi_dataset), desc="Tracks", position=0):
        '''
        Track is now a set of tokens in the llama-token space. We want to convert
        this to pseudo-english since our huggingface inference endpoint does not 
        support passing tokens directly. Essentially, we're detokenizing since 
        they'll tokenize anyway.
        '''
        correct = 0
        total = 0

        generated_tokens = track[0: CONTEXT_LEN]

        inner_bar = tqdm(range(0, len(track) - CONTEXT_LEN - 1), desc=f"Track {index}", position=1, leave=True)
        for idx in inner_bar:

            #attempting to use largest context hugging face endpoint will allow
            context = track[idx: idx + CONTEXT_LEN]
            true_next_token = track[idx+CONTEXT_LEN]
            #logger.debug(f"Context from track: {context}, length of context is {len(context)}")

            pseudo_english = llama_tok.detokenize(context)
            #logger.debug(f"Generated pseudo_english prompt: {pseudo_english}")

            out, tokens = llama.infer(pseudo_english)
            token_out = tokens[0]

            while token_out.id - 10000 > midi_dataset.max_tokens:
                out, tokens = llama.infer(pseudo_english, nextResponse=True)
                token_out = tokens[0]

            generated_tokens.append(token_out.id)

            if token_out.id == true_next_token:
                #logger.debug(f"Llama returned next tokens: {tokens}, true next token is {true_next_token}")
                correct += 1
            total += 1
            #logger.info(f"Current Accuracy: {correct / total}")
            inner_bar.set_postfix_str(f"Acc: {correct/total:.4f}")

        midi_dataset.__setitem__(generated_tokens)

    # Generates midi files from tokens
    midi_dataset.generate_midi_files()

    print(midi_dataset)