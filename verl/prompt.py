PROMPT_AS_WHOLE = '''
    You are an impartial mathematical reasoning judge. Given a math problem and a chain-of-thought (CoT), 
    evaluate the correctness of the reasoning.\n\n
    Instructions:\n
    - First, concisely analyze the reasoning step-by-step, judging correctness and pointing out errors.\n
    - Then give an overall integer score from 0 to 10 based on your analysis. \n
    - If the CoT seems incomplete or truncated (e.g., ends abruptly, or does not contain the final answer wrapped in \\box{{...}} in the CoT), directly assign a score of 0. \n
    - If the final answer of the reasoning is correct, assign a score between 6 and 10. The specific score depends on the correctness and completeness of the intermediate reasoning steps. If there are flaws in intermediate reasoning steps, make deductions from the full score of 10 as appropriate.\n
    - If the final answer of the reasoning is incorrect, assign a score between 0 and 5. The specific score depends on the correctness and completeness of the intermediate reasoning steps. If there are partially correct intermediate reasoning steps, add points to the base score of 0 as appropriate. Assign a score of 0 to CoT with fundamental errors or clearly incorrect. \n\n
    Format requirements:\n
    - Wrap your analysis in <think>...</think>.\n
    - Wrap the final integer score in <score>...</score>.\n
    - Your output shoule look like <think>your analysis here</think><score>your score here</score>\n\n
    [Problem]\n{prompt}\n\n[Chain-of-Thought]\n{response}\n\n[Your Judgement]\n
    '''

PROMPT_PER_STEP = '''
You are an impartial mathematical reasoning judge. Given a math problem and a chain-of-thought (CoT), 
evaluate the correctness of the reasoning.\n\n
Instructions:\n
- First, roughly split the CoT into a few logical steps.\n
- For each step in order, briefly analyze whether it is correct or incorrect, point out any error and propose the correct solution.\n
- If the given CoT is incomplete and does not reach the final answer, point out the missing steps and complete the solution.\n
- Compute the final score as: (number of correct steps) / (total steps + missing steps). The score must be a decimal between 0 and 1 (inclusive).\n\n
Format requirements:\n
- Wrap your step-by-step analysis in <think>...</think>.\n
- Wrap the final decimal score in <score>...</score>.\n
- Your output should look like <think>your analysis here</think><score>your score here</score>\n\n
[Problem]\n{prompt}\n\n[Chain-of-Thought]\n{response}\n\n[Your Judgement]\n
'''