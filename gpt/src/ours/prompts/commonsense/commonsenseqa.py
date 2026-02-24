cot_prompt = '''
Given the question, output the rationale step by step and give the final answer. You should choose the best answer.

Question: Sammy wanted to go to where the people were. Where might he go?  
Answer Choices : (a) race track  (b) populated area  (c) the desert  (d) apartment  (e) roadblock  
Answer:  
Fact: 
Sammy wanted to go to a place where many people are.\n
Race tracks and apartments may not always have people.\n 
The desert and roadblocks generally have few or no people.\n 
Populated areas are places known to have many people.\n
Reasoning:
Therefore, the best place for Sammy to go is a populated area.\n
### So the answer is : (b) populated area

Question: The fox walked from the city into the forest, what was it looking for?
Answer Choices : (a) pretty flowers  (b) hen house  (c) natural habitat  (d) storybook
Answer:
Fact:
The forest is a natural environment for wild animals like foxes.\n
Hen houses and storybooks are man-made and not typically found in forests.\n
Pretty flowers are unrelated to the fox’s needs.\n
Natural habitat refers to a place where a species normally lives.\n
Reasoning:
Therefore, the fox was most likely looking for its natural habitat.\n
### So the answer is : (c) natural habitat

Question: Google Maps and other highway and street GPS services have replaced what?
Answer Choices : (a) United States  (b) Mexico  (c) countryside  (d) atlas
Answer:
Fact:
Google Maps and GPS services are used for navigation and finding directions.\n
The United States and Mexico are countries, and the countryside is a type of location, not a tool for navigation.\n
An atlas is a book of maps that was traditionally used for navigation.\n
Reasoning:
Therefore, the correct answer is atlas, as it was used for the same purpose before GPS existed.\n
### So the answer is : (d) atlas

Question: {question}
Answer Choices: {options}
Answer:
'''

value_prompt = '''
Provide a "Yes" or "No" response on whether the reasoning for the following question is correct. 
Answer with exactly “Yes” or “No”, no other words or explanation.

Example 1
Question: Sammy wanted to go to where the people were. Where might he go?
Reasoning:
The answer must be a place with a lot of people.\n
Of the above choices, only populated areas have a lot of people.\n
Answer: Yes

Example 2
Question: Google Maps and other highway and street GPS services have replaced what?
Reasoning:
The answer must be something that used to do what Google Maps and GPS services do, which is to give directions.\n
Of the above choices, only atlases are used to give directions.\n
Answer: Yes

Question: {question}
Reasoning:
{output}
Answer:
'''