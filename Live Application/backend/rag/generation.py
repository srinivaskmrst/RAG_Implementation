"""
Generation: the final answer from the retrieved chunks (the "G" in RAG).
Uses the dataset's answer prompt (rag/prompts.py) - for the bank the guarded system prompt.
"""


def generate_answer(run, question, chunks):
    context = "\n\n".join(f"[Source {n}] {c['label']}:\n{c.get('evidence') or c['text'].split(': ', 1)[-1]}"
                          for n, c in enumerate(chunks, start=1))
    prompt = f"""{run.prompts["answer"]}

{context}

Question: {question}"""
    answer = run.llm(prompt)
    run.step("Generate answer", f"{run.label} with {len(chunks)} sources", kind="llm")
    return answer
