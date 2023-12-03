from huggingface_hub import InferenceClient
import omegaconf

class LLM:
    '''
    An abstraction for an LLM running on hugging face hub text-generation 
    inferecen endpoint.
    '''
    def __init__(self, config):
        api_config = omegaconf.OmegaConf.load(config)
        self.client = InferenceClient(model=api_config.endpoint_url, token=api_config.api_key)
    
    def infer(self, prompt, nextResponse=False):
        '''
        Performs the actual inference on hugging face endpoint, returning
        a generated string and a list of tokens and information about them.
        '''
        out = self.client.text_generation(prompt=prompt, details=True, do_sample=nextResponse, max_new_tokens=1)
        return out.generated_text, out.details.tokens