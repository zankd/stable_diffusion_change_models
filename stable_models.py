import json
import requests
import io, re, os
import base64
from PIL import Image

# Set the URL 
url = "http://127.0.0.1:7860"

#api_txt2img = /sdapi/v1/txt2img
#api_img2img = /sdapi/v1/img2img
#api_url = /sdapi/v1/sd-models
#api_url = /sdapi/v1/sd-vae
#api_url = /sdapi/v1/hypernetworks
# loras = /sdapi/v1/loras
# /sdapi/v1/embeddings

#api_url = /sdapi/v1/styles
#api_url = /sdapi/v1/samplers
#api_url = /sdapi/v1/upscalers
#api_url = /sdapi/v1/realesrgan-models
#api_url = /sdapi/v1/face-restorers

#api_url = /agent-scheduler/v1/queue/txt2img
#api_url = /agent-scheduler/v1/task/{id}/run

#        ExtraSingle: "/sdapi/v1/extra-single-image",
#        ExtraBatch:  "/sdapi/v1/extra-batch-images",
#        PNGInfo:     "/sdapi/v1/png-info",
#        Progress:    "/sdapi/v1/progress",
#        Interrogate: "/sdapi/v1/interrogate",
#        Interrupt:   "/sdapi/v1/interrupt",
#        Skip:        "/sdapi/v1/skip",
#        Options:     "/sdapi/v1/options"

#api_url = /sdapi/v1/scripts

"""
{
  "txt2img": [
    "prompt matrix",
    "prompts from file or textbox",
    "refiner",
    "seed",
    "x/y/z plot",
    "stylepile",
    "adetailer",
    "dynamic prompts v2.16.3",
    "reactor",
    "depthmap",
    "extra options"
  ],
  "img2img": [
    "img2img alternative test",
    "loopback",
    "outpainting mk2",
    "poor man's outpainting",
    "prompt matrix",
    "prompts from file or textbox",
    "refiner",
    "sd upscale",
    "seed",
    "x/y/z plot",
    "stylepile",
    "adetailer",
    "dynamic prompts v2.16.3",
    "reactor",
    "depthmap",
    "extra options"
  ]
}
"""

# Define the new payload in JSON format
new_payload = {
    "prompt": "a __topanimals__  driving a __vehicle__ with {playful|excited|angry} face expression during  __daytime__ in  __backscenes__ scene in  {summer|winter} highly detailed, intricate, ultra HD, sharp photo, professional studio lighting, in focus, (Best Quality, Masterpiece:1.4), (Realism:1.2), (Absurdres:1.2), (photorealistic:1.3)  <lora:add_detail:1>",
    "negative_prompt": "(worst quality:1.2), (bad quality:1.2), (poor quality:1.2), low res, ((monochrome)), ((grayscale)), text, error, cropped, jpeg artifacts, signature, watermark, username, blurry, deformed, double, duplicate, ((out of frame)), bad art, disfigured, ugly, unrealistic, (poorly drawn), boring, sketch, lackluster, horrific, b&w, crooked, broken, weird, odd, distorted, drawing, painting, crayon, sketch, graphite, impressionist, noisy, soft, grainy",
    "styles": ["string"],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "DPM++ 2M SDE Karras",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 9,
    "width": 648,
    "height": 896,
    "restore_faces": True,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "eta": 0,
    "denoising_strength": 0.7,
    "s_min_uncond": 0,
    "enable_hr": True,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 1,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 20,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_prompt": "",
    "hr_negative_prompt": "",
    "alwayson_scripts": {
        "Dynamic Prompts v2.17.1": {
            "args": [
                True,
                True,
                1,
                False,
                False,
                False,
                1.1,
                1.5,
                100,
                0.8,
                False,
                False,
                True,
                False,
                False,
                0,
                "Gustavosta/MagicPrompt-Stable-Diffusion",
                None
            ]
        }
    }
}

#    "script_name": "Ultimate SD upscale",
#    "script_args": ["",512,0,8,32,64,0.275,32,3,false,0,true,8,3,2,1080,1440,1.875],

#    "script_name": "SD upscale",
#    "script_args": ["null", 64, "R-ESRGAN 4x+", 1.5],

"""

    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 0,
    "override_settings": None,
    "override_settings_restore_afterwards": True,
    "refiner_checkpoint": None,
    "refiner_switch_at": None,
    "disable_extra_networks": False,
    "comments": None,    
    "sampler_index": "Euler",
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "input_image": "LR_encoded_image (base64 image)",
                    "model": "control_v11f1e_sd15_tile [a371b31b]",
                    "resize_mode":1,
                    "control_mode":0,
                    "weight":1,
                },
                {
                    "input_image": "Depth_encoded_image (base64 image)",
                    "model": "control_v11f1p_sd15_depth [cfd03158]",
                    "resize_mode":1,
                    "control_mode":0,
                    "weight":1,
                }
            ]
        }
    }

"""

def clean_filename(seed, prompt, index):
    cleaned_prompt = re.sub(r'\W+', '', prompt)[:50]
    return f"{seed}_{cleaned_prompt}_{index}.png"

def generate_unique_filename(seed, prompt, index, model_name=""):
    cleaned_prompt = re.sub(r'\W+', '', prompt)[:50]
    filename = f"{seed}_{cleaned_prompt}_{index}"
    if model_name:
        filename = f"{filename}_{model_name}"

    return f"{filename}.png"


def get_model_from_history():
    try:
        response = requests.get(url=f'{url}/agent-scheduler/v1/history?limit=1&offset=0') 

        if response.status_code == 200:
            history_data = response.json()
            tasks = history_data.get('tasks', [])

            if tasks:
                # Extract model information from the latest task
                latest_task = tasks[0]
                result_info = json.loads(latest_task.get('result', '{}'))
                model_name = result_info.get('infotexts', [])[0].split('Model: ')[1].split(',')[0]
                print(f"Model used in the latest task: {model_name}")

                return model_name
            else:
                print("No tasks found in the history data.")

        else:
            print(f"Failed to fetch history data. Status code: {response.status_code}")

    except requests.RequestException as e:
        print(f"Request failed: {e}")

def generate_images():
    try:
        print("Connected OK")
        print("Sending payload:")
        print(json.dumps(new_payload, indent=2))

        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=new_payload)
        print("Payload sent")

        # Check if the response is successful
        if response.status_code == 200:
            r = response.json()

            # Extract the seed from the 'info' section of the response
            seed = json.loads(r.get('info', '{}')).get('seed', None)

            # Check if the seed is available
            if seed is not None and seed != -1:
                print(f"Generated seed: {seed}")
            else:
                # Use the seed from the payload if it's not available in the response
                seed = new_payload.get('seed', None)
                if seed is not None and seed != -1:
                    print(f"Using seed from payload: {seed}")
                else:
                    print("Seed not found in the response or payload.")

            image_data_list = r.get('images', [])

            os.makedirs("images", exist_ok=True)

            available_models = get_available_models()

            for index, image_data in enumerate(image_data_list):
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                filename = generate_unique_filename(seed, new_payload['prompt'], index)

                # Set the default model name if no models are available
                model_name = 'UnknownModel'

                if available_models:
                    # Extract the first model's checkpoint name
                    model_name = available_models[0].get('sd_model_checkpoint', 'UnknownModel')

                image.save(f'images/{filename}_{model_name}.png')
                print(f"Image '{filename}_{model_name}.png' saved.")

                # Extract additional details from new_payload
                steps = new_payload.get('steps', 'N/A')
                sampler_name = new_payload.get('sampler_name', 'N/A')
                cfg_scale = new_payload.get('cfg_scale', 'N/A')
                width = new_payload.get('width', 'N/A')
                height = new_payload.get('height', 'N/A')
                denoising_strength = new_payload.get('denoising_strength', 'N/A')
                enable_hr = new_payload.get('enable_hr', 'N/A')
                hr_upscaler = new_payload.get('hr_upscaler', 'N/A')
                hr_second_pass_steps = new_payload.get('hr_second_pass_steps', 'N/A')

                # Extract lora_hashes from the payload
                lora_hashes = "N/A"
                alwayson_scripts = new_payload.get('alwayson_scripts', {})
                simple_wildcards = alwayson_scripts.get('Simple wildcards', {})
                if 'hashes' in simple_wildcards:
                    lora_hashes = simple_wildcards['hashes']

                # Create a text file with image details
                with open(f'images/{filename}_{model_name}.txt', 'w') as file:
                    file.write(f"Image Details for {filename}:\n\n")
                    file.write(f"Prompt: {new_payload['prompt']}\n\n")
                    file.write(f"Negative Prompt: {new_payload['negative_prompt']}\n\n")
                    file.write(f"Model Used: {model_name}\n\n")
                    file.write(f"Steps: {steps}\n")
                    file.write(f"Sampler: {sampler_name}\n")
                    file.write(f"CFG scale: {cfg_scale}\n")
                    file.write(f"Seed: {seed}\n")
                    file.write(f"Size: {width}x{height}\n")
                    file.write(f"Denoising Strength: {denoising_strength}\n")
                    file.write(f"Hires Upscale: {enable_hr}\n")
                    file.write(f"Hires Upscaler: {hr_upscaler}\n")
                    file.write(f"Hires Steps: {hr_second_pass_steps}\n")
                    file.write(f"Lora Hashes: {lora_hashes}\n\n")

        else:
            print(f"Failed to generate images. Status code: {response.status_code}")

    except requests.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Error processing the prompt: {str(e)}")

def get_available_models():
    response = requests.get(url=f'{url}/sdapi/v1/sd-models')
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to get available models.")
        return []

def switch_model(model_name):
    opt = requests.get(url=f'{url}/sdapi/v1/options')
    opt_json = opt.json()
    opt_json['sd_model_checkpoint'] = model_name
    requests.post(url=f'{url}/sdapi/v1/options', json=opt_json)

def generate_images_with_model(model_name, seed):
    switch_model(model_name)
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=new_payload)
    if response.status_code == 200:
        r = response.json()
        image_data_list = r.get('images', [])

        os.makedirs("images", exist_ok=True)

        # Extract the seed from the response
        seed = json.loads(r.get('info', '{}')).get('seed', None)

        for index, image_data in enumerate(image_data_list):
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            filename = generate_unique_filename(seed, new_payload['prompt'], index, model_name)
            image.save(f'images/{filename}.png')
            print(f"Image '{filename}.png' generated using model: {model_name}")

            # Extract additional details from new_payload
            steps = new_payload.get('steps', 'N/A')
            sampler_name = new_payload.get('sampler_name', 'N/A')
            cfg_scale = new_payload.get('cfg_scale', 'N/A')
            width = new_payload.get('width', 'N/A')
            height = new_payload.get('height', 'N/A')
            denoising_strength = new_payload.get('denoising_strength', 'N/A')
            enable_hr = new_payload.get('enable_hr', 'N/A')
            hr_upscaler = new_payload.get('hr_upscaler', 'N/A')
            hr_second_pass_steps = new_payload.get('hr_second_pass_steps', 'N/A')

            # Extract lora_hashes from the payload
            lora_hashes = "N/A"
            alwayson_scripts = new_payload.get('alwayson_scripts', {})
            simple_wildcards = alwayson_scripts.get('Simple wildcards', {})
            if 'hashes' in simple_wildcards:
                lora_hashes = simple_wildcards['hashes']

            # Create a text file with image details
            with open(f'images/{filename}_{model_name}.txt', 'w') as file:
                file.write(f"Image Details for {filename}:\n\n")
                file.write(f"Prompt: {new_payload['prompt']}\n\n")
                file.write(f"Negative Prompt: {new_payload['negative_prompt']}\n\n")
                file.write(f"Model Used: {model_name}\n\n")
                file.write(f"Steps: {steps}\n")
                file.write(f"Sampler: {sampler_name}\n")
                file.write(f"CFG scale: {cfg_scale}\n")
                file.write(f"Seed: {seed}\n")
                file.write(f"Size: {width}x{height}\n")
                file.write(f"Denoising Strength: {denoising_strength}\n")
                file.write(f"Hires Upscale: {enable_hr}\n")
                file.write(f"Hires Upscaler: {hr_upscaler}\n")
                file.write(f"Hires Steps: {hr_second_pass_steps}\n")
                file.write(f"Lora Hashes: {lora_hashes}\n\n")
    else:
        print(f"Failed to generate images using model: {model_name}. Status code: {response.status_code}")

# Main execution
try:
    generate_images()

    available_models = get_available_models()

    for model in available_models:
        model_name = model['model_name']
        generate_images_with_model(model_name, seed=new_payload['seed'])

except requests.RequestException as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"Error processing the prompt: {str(e)}")
