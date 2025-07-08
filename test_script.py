from litgpt import LLM

llm = LLM.load("/raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama/final")
while True:
    text = llm.generate(input("Input:"))
    print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.