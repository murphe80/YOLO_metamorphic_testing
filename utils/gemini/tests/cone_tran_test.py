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
prompt_rotation = "Transform the provided race track image by inserting a blue cone fallen over on its side in the foreground of the image. The cone should be consistent in the style of the other cones in the image. All other elements must remain exactly as they are: preserve the track layout, background scenery and lighting conditions. Only the orientation/facing direction of the cones should change."
prompt_fog = "Using the provided image of a race track, degrade the quality of the image by changing the weather to a foggy day with low visibility."
prompt_cone_insertion = "Using the provided image of a race track, please add **one** additional blue cone to the track in the foreground (closer to the camera/front of the image). The new cone should be clearly visible and prominent, not placed far in the background. Match the style, size, and proportions of the existing cones, ensuring it appears realistic with appropriate perspective for its foreground position. The cone should blend naturally with the lighting, shadows, and overall scene. Keep the rest of the track, including the background and existing elements, unchanged."
prompt_rain = "Using the provided image of a race track, degrade the quality of the image by changing the weather to a wet, rainy day with poor lighting conditions. Add appropriate distortion as a result of rain drops landing on the camera."
prompt_occlusion = "Using the provided image of a race track, insert a blue cone in the foreground of the image infront of an existing cone. The inserted cone should occlude at least half of an existing cone. The cone should be consistent in the style of the other cones in the image. All other elements must remain exactly as they are: preserve the track layout, background scenery and lighting conditions. Only the orientation/facing direction of the cones should change."
prompt_heat_distortion = "Using the provided image of a race track, apply a heavy analog video distortion effect to the image. Overlay distinct, dark horizontal scanlines across the entire frame. Introduce a wavy, rippling heat haze distortion that slightly warps the image content."


# test request for two files 
with open('amz_00805.png', 'rb') as img_file:
    image_1 = base64.b64encode(img_file.read()).decode('utf-8')
with open('ff_00006.png', 'rb') as img_file:
    image_2 = base64.b64encode(img_file.read()).decode('utf-8')
test_request = [
    {
        'contents': [
            {
                'parts': [
                    {
                        'inline_data': {
                            'mime_type': 'image/png',
                            'data': image_1
                        }
                    },
                    {
                        'text': prompt_heat_distortion
                    }
                ]
            }
        ],
        'config': {
            'response_modalities': ['TEXT', 'IMAGE']
        }
    },
    {
        'contents': [
            {
                'parts': [
                    {
                        'inline_data': {
                            'mime_type': 'image/png',
                            'data': image_2
                        }
                    },
                    {
                        'text': prompt_heat_distortion
                    }
                ]
            }
        ],
        'config': {
            'response_modalities': ['TEXT', 'IMAGE']
        }
    }
]

inline_batch_job = client.batches.create(
    model="models/gemini-2.5-flash-image",
    src=test_request,
    config={
        'display_name': "test_insertion-job-1",
    },
    
)

# 3. Monitor job status
job_name = inline_batch_job.name
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


if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
    print("Results are inline:")
    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
        print(f"Response {i+1}:")
        if inline_response.response:
            for part in inline_response.response.candidates[0].content.parts:
                if part.text:
                    print(part.text)
                elif part.inline_data:
                    print(f"Image mime type: {part.inline_data.mime_type}")
                    image = part.as_image()
                    image.save(f"image_cone_heat_distortion_{i+1}.png")
        elif inline_response.error:
            print(f"Error: {inline_response.error}")
elif batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")