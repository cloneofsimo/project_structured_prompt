
import openai
import pandas as pd
from openai import OpenAI


import pandas as pd
import concurrent.futures
import time
import requests
from openai import OpenAI

import re



TASK_TEMPLATE = """
{task_description}

Example:
{examplars}

Based on the above example, generate the answer for following query:
Query:{query}\n

Tell me your thought process to derive the answer. Make sure that at the end of the generation, rewrite the answer in following format:
'\nAnswer:\n\nYOUR ANSWER' 
"""


def generate_single(task_description, examplars, query):
    req = TASK_TEMPLATE.format(
        task_description=task_description, examplars=examplars, query=query
    )
    return req


def parse_gpt_response_with_regex(response):
    """
    Parses the GPT-4 response using regex to extract the answer after 'Answer:\n'.
    """
    # Regex pattern to find text following 'Answer:\n'
    match = re.search(r"Answer:\n(.*)", response, re.DOTALL)

    # Extracting and returning the matched group if found
    if match:
        return match.group(1).strip()
    else:
        return "No specific answer found."


api_key = ""




def call_api_with_retry(td, examplars_as_str, query, max_retries=3):
    """Function to call the API with retries on network failure."""
    client = OpenAI(api_key=api_key)
    retries = 0

    while retries < max_retries:
        try:
            request = generate_single(td, examplars_as_str, query)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": request}],
            )
            total_mes = response.choices[0].message.content
            mes = parse_gpt_response_with_regex(total_mes)
            return mes, total_mes
        except requests.exceptions.RequestException:
            retries += 1
            time.sleep(1)  # Wait for 1 second before retrying

    return None, None  # Return None if all retries fail


def process_query(td, examplars_as_str, query):
    """Helper function for processing a single query."""
    return call_api_with_retry(td, examplars_as_str, query)
