'''
We have to use the llama tokenizer/detokenizer to configure inputs to the llama model.
Even though we have a midi tokenizer, we need to given prompts to llama in english, 
so, the pipeline is going to look like:
MIDI -> Midi Tokens -> Detokenize with LLama -> Garbage that resembles english -> llama -> tokenize with llama -> compare
'''
from transformers import AutoTokenizer
import omegaconf

import logging
logging.basicConfig()
logger = logging.getLogger("tok_utils")
logger.setLevel(logging.DEBUG)

class Llama_Tokenizer:
    def __init__(self, config_file):
        config = omegaconf.OmegaConf.load(config_file)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast = True, token=config.api_key)

    def tokenize(self, english):
        '''
        Returns a list of ids.
        '''
        return self.tokenizer(english, add_special_tokens=False)
    
    def detokenize(self, tokens):
        '''
        Takes in a list of tokens and returns english.
        '''
        return self.tokenizer.decode(tokens)

if __name__ == "__main__":
    llama_tok = Llama_Tokenizer("config.yaml")
    x = llama_tok.tokenize('This is a sentence')
    logger.debug(f"Tokenized version of \"This is a sentence\": {x}")
    logger.debug(f"Detokenized version of that: {llama_tok.detokenize(x["input_ids"])}")