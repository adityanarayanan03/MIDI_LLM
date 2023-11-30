from huggingface_hub import InferenceClient
import omegaconf

api_config = omegaconf.OmegaConf.load("api_config.yaml")

client = InferenceClient(model=api_config.endpoint_url, token=api_config.api_key)
out = client.text_generation(prompt="The meaning of life is ", details=True)

#We get the text generated with 
print(out.generated_text)

#We can get individual tokens using
print(out.details.tokens)