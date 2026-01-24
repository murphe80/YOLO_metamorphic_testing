# script to generate metmaorphic tests using gemini 
# Source: https://ai.google.dev/gemini-api/docs/batch-api?batch=file#image-generation 

import json
import time
import base64
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
gemini_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_key)

# 1. Read and upload file
file_name = "fog-batch-image-requests.jsonl"
with open(file_name, 'b+w') as f:
    uploaded_file = f
    uploaded_file = client.files.upload(
        file=uploaded_file,
        config=types.UploadFileConfig(display_name='batch-image-requests', mime_type='jsonl')
    )
    print(f"Uploaded file: {uploaded_file.name}")

# # 2. Create batch job
file_batch_job = client.batches.create(
    model="gemini-2.5-flash-image",
    src=uploaded_file.name,
    config={
        'display_name': "metamorphic_gen_job_1",
    },
)
print(f"Created batch job: {file_batch_job.name}")

# 3. Monitor job status
job_name = file_batch_job.name
print(f"Polling status for job: {job_name}")

completed_states = set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
])

batch_job = client.batches.get(name=job_name) # Initial get
while batch_job.state.name not in completed_states:
  print(f"Current state: {batch_job.state.name}")
  time.sleep(10) # Wait for 10 seconds before polling again
  batch_job = client.batches.get(name=job_name)

print(f"Job finished with state: {batch_job.state.name}")

# 4. Retrieve results
if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
    result_file_name = batch_job.dest.file_name
    print(f"Results are in file: {result_file_name}")
    print("Downloading result file content...")
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')
    # The result file is also a JSONL file. Parse and print each line.
    for line in file_content.splitlines():
      if line:
        parsed_response = json.loads(line)
        if 'response' in parsed_response and parsed_response['response']:
            for part in parsed_response['response']['candidates'][0]['content']['parts']:
              if part.get('text'):
                print(part['text'])
              elif part.get('inlineData'):
                print(f"Image mime type: {part['inlineData']['mimeType']}")
                data = base64.b64decode(part['inlineData']['data'])
        elif 'error' in parsed_response:
            print(f"Error: {parsed_response['error']}")
elif batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")