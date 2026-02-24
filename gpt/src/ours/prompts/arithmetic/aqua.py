# v1
cot_prompt = """
Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64
A: 
If 10 is added to each number, then the mean of the numbers also increases by 10.\n
So the new mean would be 50.\n
### The answer is (a).

Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a.
Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2
A: 
If a / b = 3/4, then b = 4a / 3.\n
So 8a + 5(4a / 3) = 22.\n
This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22.\n
So a is equal to 3/2.\n
### The answer is (b).

Q: How many keystrokes are needed to type the numbers from 1 to 500?
Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 1788
A:
There are 9 one-digit numbers from 1 to 9.\n
There are 90 two-digit numbers from 10 to 99.\n
There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.\n
### The answer is (b).

Q: {question}
Answer Choices: {options}
A:
"""
value_prompt = """
Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
Answer with exactly “Yes” or “No”, no other words or explanation.

Example 1  
Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?  
Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64  
Reasoning:
If 10 is added to each number, then the mean of the numbers also increases by 10.\n
So the new mean would be 50.\n
Answer: Yes

Example 2  
Q: If a / b = 3/4 and 8a + 5b = 22, then find the value of a.  
Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2  
Reasoning:  
If a / b = 3/4, then b = 4a / 3.\n
So 8a + 5(4a / 3) = 22.\n
This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22.\n
So a is equal to 3/2.\n
Answer: Yes

Question: {question}
Reasoning:
{output}
Answer:
"""

# v2
# cot_prompt = """
# Let’s solve this problem step by step; for each sentence, we will write what it means as a mathematical expression, and then provide a final answer.

# Example 1  
# Q: How much is 70% of 40 is greater than 4/5 of 25?
# Answer Choices: A)22, B)67, C)88, D)12, E)8
# A:
# Sentence 1: 70% of 40 is 0.7 x 40 = 28 \n
# Sentence 2: 4/5 of 25 is 25 x 4/5 = 20 \n
# Sentence 3: The difference is 28 - 20 = 8 \n

# ### So the answer is (e) 8

# Example 2  
# Q: If a / b = 3/4 and 8a + 5b = 22, then find the value of a.  
# Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2  
# A:  
# Sentence 1: If a / b = 3/4 , This means: b = 4a / 3 \n
# Sentence 2: 8a + 5b = 22 , This means: 8a + 5(4a / 3) = 22 \n
# Sentence 3: then find the value of a. , This means: 8a + (20a / 3) = 22 , (44a / 3) = 22 , a = 3/2 \n
# ### So the answer is (b) 3/2

# Example 3  
# Q: How many keystrokes are needed to type the numbers from 1 to 500?  
# Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 1788  
# A:  
# Sentence 1: How many keystrokes are needed to type the numbers from 1 to 500? , This means: (1 to 9) + (10 to 99) + (100 to 500) \n
# - 1 to 9: 9 numbers x 1 digit = 9 \n
# - 10 to 99: 90 numbers x 2 digits = 180 \n  
# - 100 to 500: 401 numbers x 3 digits = 1203 \n  
# Total = 9 + 180 + 1203 = 1392 \n
# ### So the answer is (b) 1392

# Q: {question}  
# Answer Choices: {options}  
# A:
# """

# value_prompt = """
# Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
# Answer with exactly “Yes” or “No”, no other words or explanation.

# Example 1  
# Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?  
# Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64  
# Reasoning:
# Sentence 1: John found that the average of 15 numbers is 40. , This means: mean = 40 \n
# Sentence 2: If 10 is added to each number then the mean of the numbers is? , This means: mean + 10 = 40 + 10 = 50 \n
# Answer: Yes

# Example 2  
# Q: If a / b = 3/4 and 8a + 5b = 22, then find the value of a.  
# Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2  
# Reasoning:  
# Sentence 1: If a / b = 3/4 , This means: b = 4a / 3 \n
# Sentence 2: 8a + 5b = 22 , This means: 8a + 5(4a / 3) = 22 \n
# Sentence 3: then find the value of a. , This means: 8a + (20a / 3) = 22 , (44a / 3) = 22 , a = 3/2 \n
# Answer: Yes

# Question: {question}
# Reasoning:
# {output}
# Answer:
# # """