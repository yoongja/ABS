# 5-shot
cot_prompt = '''Given the calculation steps below, write a single parenthesized arithmetic expression under "Answer:" that combines the original input numbers with +, -, *, or / to obtain 24. 
Do not provide any explanation—output only the final expression as shown in the examples.

Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + (8 / 4)) * 8 = 24

Input: {input}
Steps:
'''

cot_propose_prompt = '''You are solving the Game of 24. At each step, combine two of the remaining numbers using +, -, *, or /. Use all numbers exactly once to make 24.

Generate {k} diverse next steps:
- Use different pairs and various operations.
- Do not write any explanation—output only equations as in the examples.

Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 - 8 = 6 (left: 2 6 8)
2 * 8 = 16 (left: 8 14 16)

# Now solve this:
Input: {input}
Possible next steps:
'''

value_prompt= '''Answer "Yes" or "No" to indicate whether the following candidate equation, made by combining exactly two of the current numbers with +, -, *, or /, could still allow a solution to reach 24 using all numbers in subsequent steps.
Respond with exactly "Yes" or "No" only.

# Example 1
Input: 4 9 10 13
Current numbers: 4 9 10 13
Candidate equation:
13 - 9 = 4 (left: 4 4 10)
Answer: Yes
Candidate equation:
10 * 13 = 130 (left: 130 4 9)
Answer: No

Input: {input}
Current numbers: {current}
Candidate equation:
{candidate}
Answer:
'''

value_last_step_prompt = '''Provide a "Yes" or "No" response on whether the following equation is a correct solution for the Game of 24.
The equation must:
- Use all given numbers exactly once
- Use only +, -, *, /
- Evaluate to exactly 24

Answer with exactly “Yes” or “No”, no other words or explanation.

# Example 1
Input: 4 9 10 13
Equation:
(10 - 4) * (13 - 9) = 24
Answer: Yes

Input: {input}
Equation:
{final_equation}
Answer:
'''
