cot_prompt = '''
Q: Do hamsters provide food for any animals?
A:
Hamsters are prey animals.\n
Prey animals are food for predators.\n
Thus, hamsters provide food for some animals.\n
### So the answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A:
Brooke Shields graduated from Princeton University.\n
According to US news, Princeton University and University of Pennsylvania are ranked as the number 1 and 6 national college, respectively.\n
This can indicate that Princeton University is about as academically rigorous as the University of Pennsylvania.\n
Thus, Brooke Shields could also succeed at University of Pennsylvania.\n
### So the answer is yes.

Q: Hydrogen’s atomic number squared exceeds number of Spice Girls?
A:
Hydrogen is the first element and has an atomic number of one.\n
To square a number, you multiply it by itself, so one squared is one.\n
In comparison, the Spice Girls has five members.\n
Thus, Hydrogen’s atomic number squared is less than 5.\n
### So the answer is no.

Q: {question}
A:
'''


value_prompt = '''
Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
Answer with exactly “Yes” or “No”, no other words or explanation.

Example 1
Question: Do hamsters provide food for any animals?
Reasoning:
Hamsters are prey animals.\n
Prey animals are food for predators.\n
Thus, hamsters provide food for some animals.\n
Answer: Yes

Example 2
Question: Could Brooke Shields succeed at University of Pennsylvania?
Reasoning:
Brooke Shields graduated from Princeton University.\n
According to US news, Princeton University and University of Pennsylvania are ranked as the number 1 and 6 national college, respectively.\n
This can indicate that Princeton University is about as academically rigorous as the University of Pennsylvania.\n
Thus, Brooke Shields could also succeed at University of Pennsylvania.\n
Answer: Yes

Question: {question}
Reasoning:
{output}
Answer:
'''