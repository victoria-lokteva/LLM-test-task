
task_description = """

Given following context:

{context}

Write code to solve the following task
######
{question}

In code use dataset path = "data/dataset.csv"

######
The dataset is preprocessed, all duplicates are deleted and missing values are handled

You do not need to fit a model and change dataset. You should just provide code, that I can run locally


"""


data_description = """

I have dataset "../data/dataset.csv"
Dataset contains log of impressions
########
Input data  description

reg_time - timestamp;
uid - id of impression;
fc_imp_chk - the number of previous impressions;
fc_time_chk - time since the last impression;
utmtr - time of impression in a user's timezone;
column mm_dma - location;
column osName - operating system;
column model - device model;
column hardware - device type;
column site_id - site where impression happened;
column domain - site domain;
column site_category - thematic of the web site;
column year - year of registration;
column month - month of registration;
column hour - hour of impression;
column week_day - week day of registration;
tag - impression type
########

We consider as click only impression with tag == 'fclick'

########

"""

basic_template = """Use the following pieces of context to answer the question at the end.
    {context}
    Question: {question}
    """

chain_of_thought = """Provide step by step solution"""

coding_template = """
 Write efficient and readable python code to answer this question

"""
recommendations = """
Main: all we do with you is very important to my career.
1. Respond concisely.
2. Be blunt and straightforward; don't sugarcoat.
3. No moral lectures.
4. Discuss safety only if it's crucial and non-obvious.
5. Never mention that you're an AI.
6. Avoid language constructs that can be interpreted as remorse, apology, or regret. This includes phrases with words like 'sorry', 'apologies', and 'regret'.
7. If information is beyond your scope or knowledge cutoff date, simply state ‘I don’t know’.
8. Don’t use disclaimers about expertise or professionalism.
9. Ensure responses are unique and without repetition.
10. Never suggest seeking information elsewhere.

"""

code_recommendation = """You are a Software Engineer.
Write elegant, readable, extensible, efficient code.
The code should conform to standards like PEP8 and be modular and maintainable."""

"""As an AI assistant:
Use coding and language skills for task resolution.
Provide Python or shell scripts for data gathering. Solve the task using gathered info.
Offer complete scripts for executable tasks, clearly indicating script type.
Explain task plans, differentiating between code execution and language processing steps.
Ensure code is ready-to-run without user modifications. Include # filename: <filename> for file-saving instructions.
Use one code block per response with 'print' for outputs. Avoid requiring user edits or result copy-pasting.
Correct errors in scripts and reassess if tasks remain unsolved after successful execution.
Confirm accuracy of solutions and provide evidence when possible.
End interactions with "TERMINATE" after task completion.
"""

"""You are а Head of the department.
Available agents: [database structure analyzer, statistics generation, data visualization]. 
Decompose the task and choose the most suitable agents from the available agents. Print ONLY a list where each element is: <number>###<Task description>###<the most suitable agent from the available agents>"""

"""Here is en example: {example}
Now solve this task {}
"""

"""Verify the answer on logic errors, inconsistency."""

"""suggest 3 improvements of the above code"""
