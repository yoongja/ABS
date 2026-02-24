# v1
cot_prompt = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A:
There are 15 trees originally.\n
And there were 21 trees after some more were planted.\n
So 21 - 15 = 6 trees were planted.\n
### The answer is 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A:
There are originally 3 cars.\n
And 2 more cars arrive.\n
So there are 3 + 2 = 5 cars now.\n
### The answer is 5

Q: Leah had 32 chocolates and her sister had 10 more chocolates than her. If they ate 35, how many pieces do they have left in total?
A:
Leah had 32 chocolates, and her sister 32 + 10 = 42 chocolates.\n
So together they had 32 + 42 = 74 chocolates.\n
Then they ate 35 chocolates, so the remaining chocolates are 74 - 35 = 39 chocolates.\n
### The answer is 39

Q: {question}
A:
"""

value_prompt = '''
Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
Answer with exactly “Yes” or “No”, no other words or explanation.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Reasoning:
There are 15 trees originally.\n
And there were 21 trees after some more were planted.\n
So 21 - 15 = 6 trees were planted.\n
Answer: Yes

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Reasoning:
There are originally 3 cars.\n
And 2 more cars arrive.\n
So there are 3 + 2 = 5 cars now.\n
Anwer: Yes

Question: {question}
Reasoning:
{output}
Answer:
'''

# v2
# cot_prompt = '''
# Let’s solve this problem step by step; for each sentence, we will write what it means as a mathematical expression, and then provide a final answer.

# Example 1
# Q: Kate’s hair is half as long as Emily’s hair. Emily’s hair is 6 inches longer than Logan’s hair. If Logan’s hair is 20 inches, how many inches is Kate’s hair?
# A:
# Sentence 1: Kate’s hair is half as long as Emily’s hair. , This means: Kate_hair = 1/2 x Emily_hair \n
# Sentence 2: Emily’s hair is 6 inches longer than Logan’s hair. , This means: Emily_hair = Logan_hair + 6 \n
# Sentence 3: Logan’s hair is 20 inches. , This means: Logan_hair = 20 \n
# Sentence 4: how many inches is Kate’s hair? , This means: Emily_hair = 20 + 6 = 26, Kate_hair = 1/2 x 26 = 13 \n
# ### So the answer is 13

# Example 2
# Q: John puts $25 in his piggy bank every month for 2 years to save up for a vacation. He had to spend $400 from his piggy bank savings last week to repair his car. How many dollars are left in his piggy bank?
# A:
# Sentence 1: John puts $25 in his piggy bank every month for 2 years. , This means: Monthly_saving = 25, Months = 2 x 12 = 24,  Total_saving = 25 x 24 = 600 \n
# Sentence 2: He had to spend $400 from his piggy bank savings last week. , This means: Spent_amount = 400 \n
# Sentence 3 (Implied Question): How many dollars are left? , This means: Remaining = Total_saving - Spent_amount = 600 - 400 = 200 \n
# ### So the answer is 200

# Example 3
# Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
# A:
# Sentence 1: There were nine computers in the server room. , This means: Initial_computers = 9 \n
# Sentence 2: Five more computers were installed each day, from Monday to Thursday. , This means: Days = 4, Daily_install = 5, New_computers = 5 x 4 = 20 \n
# Sentence 3 (Implied Question): How many computers are now in the server room? , This means: Total_computers = Initial_computers + New_computers = 9 + 20 = 29 \n
# ### So the answer is 29

# Q: {question}
# A:
# '''

# value_prompt = '''
# Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
# Answer with exactly “Yes” or “No”, no other words or explanation.

# Example 1
# Question: Kate’s hair is half as long as Emily’s hair. Emily’s hair is 6 inches longer than Logan’s hair. If Logan’s hair is 20 inches, how many inches is Kate’s hair?
# Reasoning:
# Sentence 1: Kate’s hair is half as long as Emily’s hair. , This means: Kate_hair = 1/2 x Emily_hair \n
# Sentence 2: Emily’s hair is 6 inches longer than Logan’s hair. , This means: Emily_hair = Logan_hair + 6 \n
# Sentence 3: Logan’s hair is 20 inches. , This means: Logan_hair = 20 \n
# Sentence 4: how many inches is Kate’s hair? , This means: Emily_hair = 20 + 6 = 26, Kate_hair = 1/2 x 26 = 13 \n
# Answer: Yes

# Example 2
# Question: John puts $25 in his piggy bank every month for 2 years to save up for a vacation. He had to spend $400 from his piggy bank savings last week to repair his car. How many dollars are left in his piggy bank?
# Reasoning:
# Sentence 1: John puts $25 in his piggy bank every month for 2 years. , This means: Monthly_saving = 25, Months = 2 x 12 = 24,  Total_saving = 25 x 24 = 600 \n
# Sentence 2: He had to spend $400 from his piggy bank savings last week. , This means: Spent_amount = 400 \n
# Sentence 3 (Implied Question): How many dollars are left? , This means: Remaining = Total_saving - Spent_amount = 600 - 400 = 200 \n
# Anwer: Yes

# Question: {question}
# Reasoning:
# {output}
# Answer:
# '''

