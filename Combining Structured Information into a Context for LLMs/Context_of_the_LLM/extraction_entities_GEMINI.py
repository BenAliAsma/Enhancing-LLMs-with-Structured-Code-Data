from src.config import commit, date, version, repo_name, problem_stmt
from typing import List
import google.generativeai as genai

def evaluate_with_gemini(input_passage: str) -> List[tuple]:

    genai.configure(api_key="AIzaSyChAaSLmuQbvV9ODS_B8wC5ChPtpxd-nN4")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    prompt = f"""
System Message: You are a helpful information extraction system.
Prompt: Given a passage, your task is to extract all entities and identify their entity types.
The output should be a list of tuples of the following format:
[("function "), ... ].
Passage: {input_passage}"""

    response = model.generate_content(contents= prompt)
    try:
        output = eval(response.text.strip())
        if isinstance(output, list):
            return output
        else:
            return []
    except:
        return []

# --- Usage Example ---

if __name__ == "__main__":

    gemini_results = evaluate_with_gemini(problem_stmt)
    print("\nGemini Extracted Entities:")
    for ent in gemini_results:
       print(f"{ent[0]:<25} | {ent[1]}")

