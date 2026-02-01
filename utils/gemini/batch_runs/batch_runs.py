import json
import time
import base64
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import os

IMAGE_DIRECTORY="../../../data/base_images/test/images"

# Load Gemini API key
load_dotenv(find_dotenv())
gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

# Prompts
prompt_rotation = "Transform the provided race track image by inserting a blue cone fallen over on its side in the foreground of the image. The cone should be consistent in the style of the other cones in the image. All other elements must remain exactly as they are: preserve the track layout, background scenery and lighting conditions. Only the orientation/facing direction of the cones should change."
prompt_fog = "Using the provided image of a race track, degrade the quality of the image by changing the weather to a foggy day with low visibility."
prompt_cone_insertion = "Using the provided image of a race track, please add **one** additional blue cone to the track in the foreground (closer to the camera/front of the image). The new cone should be clearly visible and prominent, not placed far in the background. Match the style, size, and proportions of the existing cones, ensuring it appears realistic with appropriate perspective for its foreground position. The cone should blend naturally with the lighting, shadows, and overall scene. Keep the rest of the track, including the background and existing elements, unchanged."
prompt_rain = "Using the provided image of a race track, degrade the quality of the image by changing the weather to a wet, rainy day with poor lighting conditions. Add appropriate distortion as a result of rain drops landing on the camera."
prompt_occlusion = "Using the provided image of a race track, insert a blue cone in the foreground of the image infront of an existing cone. The inserted cone should occlude at least half of an existing cone. The cone should be consistent in the style of the other cones in the image. All other elements must remain exactly as they are: preserve the track layout, background scenery and lighting conditions. Only the orientation/facing direction of the cones should change."
prompt_heat_distortion = (
    "Using the provided image of a race track, apply a heavy analog video distortion effect to the image. "
    "Overlay distinct, dark horizontal scanlines across the entire frame. Introduce a wavy, rippling heat haze distortion that slightly warps the image content."
)

# Function to generate test_request from a directory
def create_test_request(image_dir: str, prompt: str):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
    test_request = []
    filenames = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        request_item = {
            'contents': [
                {
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': 'image/png',
                                'data': image_data
                            }
                        },
                        {
                            'text': prompt
                        }
                    ]
                }
            ],
            'config': {
                'response_modalities': ['TEXT', 'IMAGE']
            }
        }
        test_request.append(request_item)
        filenames.append(image_file)
    
    return test_request, filenames

test_request, filenames = create_test_request(IMAGE_DIRECTORY, prompt_rain)

# Create batch job
inline_batch_job = client.batches.create(
    model="models/gemini-2.5-flash-image",
    src=test_request,
    config={
        'display_name': "batch_transformation_job",
    },
)

# Monitor job status
job_name = inline_batch_job.name
print(f"Polling status for job: {job_name}")

completed_states = {
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
}

batch_job = client.batches.get(name=job_name)
while batch_job.state.name not in completed_states:
    print(f"Current state: {batch_job.state.name}")
    time.sleep(10)
    batch_job = client.batches.get(name=job_name)

print(f"Job finished with state: {batch_job.state.name}")

# Process results using original filenames
if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
    print("Results are inline:")
    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
        original_name = filenames[i].rsplit('.', 1)[0]  # remove extension
        print(f"Response {i+1} (from {filenames[i]}):")
        if inline_response.response:
            for part in inline_response.response.candidates[0].content.parts:
                if part.text:
                    print(part.text)
                elif part.inline_data:
                    print(f"Image mime type: {part.inline_data.mime_type}")
                    image = part.as_image()
                    output_name = f"{original_name}_cone_heat_distortion.png"
                    image.save(output_name)
        elif inline_response.error:
            print(f"Error: {inline_response.error}")
elif batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")
