# v1
cot_prompt = """
Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
A:
Yesterday was 04/30/2021.\n
One day after 04/30/2021 is 05/01/2021.\n
### Final Answer: 05/01/2021.

Q: Yesterday was April 30, 2021. What is the date tomorrow in MM/DD/YYYY?
A:
Yesterday was 04/30/2021.\n
Today is 05/01/2021.\n
One day after today (05/01/2021) is 05/02/2021.\n
### Final Answer: 05/02/2021.

Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date one week ago from today in MM/DD/YYYY?
A: 
The first day of 2019 is 01/01/2019, which is a Tuesday.\n
Today is the first Monday of 2019, which is 01/07/2019.\n
One week ago from 01/07/2019 is 12/31/2018.\n
### Final Answer: 12/31/2018.

Q: {question}
A:
"""

value_prompt = """
Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
Answer with exactly "Yes" or "No", no other words or explanation.

Question: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?  
Reasoning:
Yesterday was 04/30/2021.\n
One day after 04/30/2021 is 05/01/2021.\n
Answer: Yes

Question: Yesterday was April 30, 2021. What is the date tomorrow in MM/DD/YYYY?  
Reasoning: 
Yesterday was 04/30/2021.\n
Today is 05/01/2021.\n
One day after today (05/01/2021) is 05/02/2021.\n
Answer: Yes

Question: {question}
Reasoning:
{output}
Answer:
"""

#v2
# cot_prompt = """
# Let’s reason about the date carefully step-by-step.

# Date Arithmetic Rules:
# If the question mentions "yesterday was MM/DD/YYYY" and asks about the day after, then:
# -"Today" is one day after the given date.
# -"The day after" refers to today + 1, which is two days after the original date.

# Month Day Counts (non-leap year):
# - January: 31 days
# - February: 28 days
# - March: 31 days
# - April: 30 days
# - May: 31 days
# - June: 30 days
# - July: 31 days
# - August: 31 days
# - September: 30 days
# - October: 31 days
# - November: 30 days
# - December: 31 days

# Leap Year Rule (for February):
# - If the year is divisible by 4 and not by 100, or divisible by 400, then February has 29 days.

# Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?  
# A: 
# If 2015 is coming in 36 hours, then it is coming in 2 days. \n
# 2 days before 01/01/2015 is 12/30/2014, so today is 12/30/2014. \n
# So one week from today will be 01/05/2015. \n
# ### So the answer is 01/05/2015.

# Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?  
# A: 
# If the first day of 2019 was Tuesday, then 01/01/2019 was a Tuesday.\n
# Today is the first monday, would be six days later.\n
# So today is 01/07/2019.\n
# ### So the answer is 01/07/2019.

# Q: The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?  
# A: 
# One day after 06/01/1943 is 06/02/1943, so today is 06/02/1943.\n
# 10 days before today is 05/23/1943.\n
# ### So the answer is 05/23/1943.

# Q: It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?  
# A: 
# Today is 04/19/1969.\n
# 24 hours later is one day after today, which would be 04/20/1969.\n
# ### So the answer is 04/20/1969.

# Q: Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later. What is the date 24 hours later in MM/DD/YYYY?  
# A: 
# Today is 03/12/2002.\n
# So the date 24 hours later will be 03/13/2002.\n
# ### So the answer is 03/13/2002.

# Q: Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date yesterday in MM/DD/YYYY?  
# A: 
# The last day of February is the 28th, so Jane was born on 02/28/2001.\n
# Today is her 16-year old birthday, so today is 02/28/2017.\n
# So yesterday was 02/27/2017.\n
# ### So the answer is 02/27/2017.

# Q: {question}
# A:
# """

# value_prompt = """
# Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
# Answer with exactly "Yes" or "No", no other words or explanation.

# Example 1
# Question: Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date yesterday in MM/DD/YYYY?  
# Reasoning:
# The last day of February is the 28th, so Jane was born on 02/28/2001.\n
# Today is her 16-year old birthday, so today is 02/28/2017.\n
# So yesterday was 02/27/2017.\n
# Answer: Yes

# Example 2
# Question: It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?  
# Reasoning: 
# Today is 04/19/1969.\n
# 24 hours later is one day after today, which would be 04/20/1969.\n
# Answer: Yes

# Question: {question}
# Reasoning:
# {output}
# Answer:
# """