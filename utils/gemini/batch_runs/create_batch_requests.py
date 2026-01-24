# script to generate a json file for batch requests
import json 
import os 
import base64

file_name = "rain-batch-image-requests.jsonl"

source_images = "/home/murphe80/metamorphic_testing/YOLO_metamorphic_testing/data/base_images/test/images"

prompt_rotation = "Transform the provided race track image by inserting a blue cone fallen over on its side in the foreground of the image. The cone should be consistent in the style of the other cones in the image. All other elements must remain exactly as they are: preserve the track layout, background scenery and lighting conditions. Only the orientation/facing direction of the cones should change."
prompt_fog = "Using the provided image of a race track, degrade the quality of the image by changing the weather to a foggy day with low visibility."
prompt_cone_insertion = "Using the provided image of a race track, please add **one** additional blue cone to the track in the foreground (closer to the camera/front of the image). The new cone should be clearly visible and prominent, not placed far in the background. Match the style, size, and proportions of the existing cones, ensuring it appears realistic with appropriate perspective for its foreground position. The cone should blend naturally with the lighting, shadows, and overall scene. Keep the rest of the track, including the background and existing elements, unchanged."
prompt_rain = "Using the provided image of a race track, degrade the quality of the image by changing the weather to a wet, rainy day with poor lighting conditions. Add appropriate distortion as a result of rain drops landing on the camera."

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(source_images) 
               if f.lower().endswith(image_extensions)]
requests = []

def get_mime_type(filename):
    ext = filename.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    return mime_types.get(ext, 'image/jpeg')

with open(file_name, "w") as f:
    # for each image, create request
    for idx, image in enumerate(image_files):
        image_path = os.path.join(source_images, image)
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        mime_type = get_mime_type(image)
        
        request = {
            "key": f"img-{idx}",
            "request": {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_data
                                }
                            },
                            {
                                "text": prompt_rain
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "responseModalities": ["TEXT", "IMAGE"]
                }
            }
        }
        requests.append(request)
        
        
    for req in requests:
        f.write(json.dumps(req) + "\n")