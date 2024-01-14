
data_description = """
Dataset contains log of impressions
########
Input data  description

reg_time - timestamp;
uid - id of impression;
fc_imp_chk - the number of previous impressions;
fc_time_chk - time since the last impression;
utmtr - time of impression in a user's timezone;
column mm_dma - Designated Market Area;
column osName - operating system;
column model - device model;
column <hardware> - device type;
column <site_id> - site w here impression happened;
tag - impression type
########

We consider as click only impression with tag == 'fclick'

########

"""


"""provide step by step solution"""

with_context = """Answer the question based only on the following context:
{context}

Question: {question}
"""


"""
Analyze provided dataset and answer the question:
 "{question}"
 
Use python programming language. Write efficient and readable code, 

"""

"""You are a Software Engineer.
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