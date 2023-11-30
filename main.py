from inference import LLM

llm = LLM("config.yaml")

#Mock scenario -- generating only one token at a time
context = "I would walk 500 miles and "
for _ in range(100):
    text, tokens = llm.infer(context)
    context += text
    print(context)