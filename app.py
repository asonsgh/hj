import requests
import json
from pprint import pprint
import g4f

def fetch_imagedescription_and_script(prompt):
    # Create a response using g4f
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts. Always respond in English."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.3,
        max_tokens=2000,
        top_p=1,
        stream=False
    )

    # Extract data from the response
    try:
        output = json.loads(response.strip())
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        print("Raw response:", response)
        return [], []

    pprint(output)
    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]

    return image_prompts, texts

import os
import io
import requests
from PIL import Image
import random

def generate_images(prompts, fname):
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"

    headers = {'x-api-key': segmind_apikey}

    # Create a folder for the UUID if it doesn't exist
    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)

    currentseed = random.randint(1, 1000000)
    print ("seed ",currentseed)

    for i, prompt in enumerate(prompts):

        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        data = {
            "prompt": final_prompt,
            "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
            "style": "hdr",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 30,
            "guidance_scale": 8,
            "strength": 1,
            "seed": currentseed,
            "img_width": 1024,
            "img_height": 1024,
            "refiner": "yes",
            "base64": False
                  }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))

            image_filename = os.path.join(fname, f"{i + 1}.jpg")
            image.save(image_filename)

            print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
        else:
            print (response.text)
            print(f"Error: Failed to retrieve or save image {i + 1}")    


import requests
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os
import cv2
import numpy as np
import ffmpeg

def generate_and_save_audio(text, foldername, filename, voice_id, elevenlabs_apikey, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_apikey
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        print(response.text)
    else:
        file_path = f"{foldername}/{filename}.mp3"
        with open(file_path, 'wb') as f:
            f.write(response.content)



def create_combined_video_audio(mp3_folder, output_filename, output_resolution=(1080, 1920), fps=24):
    mp3_files = sorted([file for file in os.listdir(mp3_folder) if file.endswith(".mp3")])
    mp3_files = sorted(mp3_files, key=lambda x: int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for mp3_file in mp3_files:
        audio_clip = AudioFileClip(os.path.join(mp3_folder, mp3_file))
        audio_clips.append(audio_clip)

        # Load the corresponding image for each mp3 and set its duration to match the mp3's duration
        img_path = os.path.join(mp3_folder, f"{mp3_file.split('.')[0]}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

        # Resize the original image to 1080x1080
        image_resized = cv2.resize(image, (1080, 1080))

        # Blur the image
        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        # Overlay the original image on the blurred one
        y_offset = (output_resolution[1] - 1080) // 2
        blurred_img[y_offset:y_offset+1080, :] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = final_video.with_audio(final_audio)
    finalpath = mp3_folder+"/"+output_filename

    final_video.write_videofile(finalpath, fps=fps, codec='libx264',audio_codec="aac")

def extract_audio_from_video(outvideo):
    """
    Extract audio from a video file and save it as an MP3 file.

    :param output_video_file: Path to the video file.
    :return: Path to the generated audio file.
    """

    audiofilename = outvideo.replace(".mp4",'.mp3')

    # Create the ffmpeg input stream
    input_stream = ffmpeg.input(outvideo)

    # Extract the audio stream from the input stream
    audio = input_stream.audio

    # Save the audio stream as an MP3 file
    output_stream = ffmpeg.output(audio, audiofilename)

    # Overwrite output file if it already exists
    output_stream = ffmpeg.overwrite_output(output_stream)

    ffmpeg.run(output_stream)

    return audiofilename

# Function to generate text clips
def generate_text_clip(word, start, end, video):
    txt_clip = (TextClip(word,font_size=80,color='white',font = "Nimbus-Sans-Bold",stroke_width=3, stroke_color='black').with_position('center')
               .with_duration(end - start))

    return txt_clip.with_start(start)


def get_word_level_timestamps(model,audioname):
  segments, info = model.transcribe(audioname, word_timestamps=True)
  segments = list(segments)  # The transcription will actually run here.
  wordlevel_info=[]
  for segment in segments:
    for word in segment.words:
      wordlevel_info.append({'word':word.word,'start':word.start,'end':word.end})
  return wordlevel_info    

from faster_whisper import WhisperModel

model_size = "base"
model = WhisperModel(model_size)


import os

def add_captions_to_video(videofilename,wordlevelcaptions):
  # Load the video file
  video = VideoFileClip(videofilename)

  # Generate a list of text clips based on timestamps
  clips = [generate_text_clip(item['word'], item['start'], item['end'], video) for item in wordlevelcaptions]

  # Overlay the text clips on the video
  final_video = CompositeVideoClip([video] + clips)

  path, old_filename = os.path.split(videofilename)

  finalvideoname = path+"/"+"final.mp4"
  # Write the result to a file
  final_video.write_videofile(finalvideoname, codec="libx264",audio_codec="aac")

  return finalvideoname


topics = [
    "Success and Achievement",
    "Morning Afformations",
    "Self-Care and Wellness",
    "Gratitude and Positivity",
    "Boost Confidence",
    "Happiness and Joy",
    "Resilience and Adversity",
    "Relationships and Connections",
    "Mindfulness and Presence",
    "Empowerment",
    "Time Management and Productivity"
]

topics_goals = {
    "Success and Achievement": "Inspire people to overcome challenges, achieve success, and celebrate their victories",
    "Morning Afformations": "Encourage viewers to start their day with a positive mindset.",
    "Self-Care and Wellness": "Offer tips and reminders for self-care practices, stress reduction, and maintaining overall well-being",
    "Gratitude and Positivity": "Emphasize gratitude and positive thinking",
    "Boost Confidence": "Help build self-confidence and self-esteem",
    "Happiness and Joy": "Help people find happiness in simple moments and enjoy life's journey",
    "Resilience and Adversity": "Help build resilience in the face of adversity",
    "Relationships and Connections": "Help build meaningful relationships, foster connections, and spread love",
    "Mindfulness and Presence": "Encourage mindfulness and being present in the moment",
    "Empowerment": "Empower viewers to take control of their lives, make positive choices, and pursue their dreams",
    "Time Management and Productivity": "Provide tips about managing time effectively, staying organized, and being productive"
}

import uuid


def create_video(inp):
  topic = inp
  goal = topics_goals[inp]
  prompt_prefix = """You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:""".format(topic,goal)

  sample_output="""
    [
        { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
        { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
        ...
    ]"""

  prompt_postinstruction="""By following these instructions, you will create an impactful {} short-form video.
  Output:""".format(topic)

  prompt = prompt_prefix + sample_output + prompt_postinstruction

  image_prompts, texts = fetch_imagedescription_and_script(prompt)

  current_uuid = uuid.uuid4()
  current_foldername = str(current_uuid)

  generate_images(image_prompts, current_foldername)

  # images = read_images_from_folder(current_foldername)

  voice_id = "pNInz6obpgDQGcFmaJgB"
  for i, text in enumerate(texts):
    output_filename= str(i + 1)
    print (output_filename)
    generate_and_save_audio(text, current_foldername, output_filename, voice_id, elevenlabsapi)

  output_filename = "combined_video.mp4"
  create_combined_video_audio(current_foldername, output_filename)
  output_video_file = current_foldername+"/"+output_filename

  return output_video_file

def add_captions(inputvideo):
  print(inputvideo)
  audiofilename = extract_audio_from_video(inputvideo)
  print (audiofilename)
  wordlevelinfo = get_word_level_timestamps(model,audiofilename)
  print (wordlevelinfo)
  finalvidpath = add_captions_to_video(inputvideo,wordlevelinfo)
  print (finalvidpath)

  return finalvidpath


import gradio as gr
!

with gr.Blocks() as demo:
  gr.Markdown("# Generate shortform videos for Youtube Shorts or Instagram Reels")
  genre = gr.Dropdown(topics,value="Success and Achievement")
  btn_create_video = gr.Button('Generate Video')
  with gr.Row():
    with gr.Column():
      video = gr.Video(format='mp4',height=720,width = 405)
    with gr.Column():
      btn_add_captions = gr.Button('Add Captions')
    with gr.Column():
      finalvideo = gr.Video(format='mp4',height=720,width = 405)

  btn_create_video.click(fn=create_video,inputs=[genre],outputs=[video])
  btn_add_captions.click(fn=add_captions,inputs=[video],outputs=[finalvideo])

demo.launch(debug=True,enable_queue=True)



please try to use Gradio base theme for Webui!






ok Try To add in this Gradio app! 
page layout in 3 colums!

first colum
user will have 2 place place holders.
to input Topic
 2nd place holder for unputing for Goal! lower to these two place holders
drop down menu to choose llm from G4F or Gimini
 for llm G4F or Gemini if Gimini then another place holder will apper for api key of Gimini in that colum  thats it! (G4f=default)

2nd colum for Video Source ! 
dopdown menu with Hercai or Segmand! 
if hecai then another dopdown menu will appear with oprions with follwing options 

*"v1" , "v2" , "v2-beta" , "v3" , "lexica" , "prodia", "simurg", "animefy", "raava", "shonin" (hercai(v3) by default)


/* Default Model; "v3" */

herc.drawImage({model:"v3",prompt:"anime girl",negative_prompt:""}).then(response => {
console.log(response.url);"
if segmand then there will no other dropdown but place holder for api key! 


3rd colum for AUDIO_source 
Audio Souce drop down menu! 
ElevenLanbs or Xtts_v2

languages and voice name for both are given below! also make sure if usr slecet elevenlabs then a place holder will appear for api key will its avaiable languages and Voices! 
else XTTS v2 with his voice and languages

xtts ve language en voice any 






ViDEO_SOURCE = "Segmind" #  @param ["Segmind",  "Hercai", "Pexels" ] {allow-input: true}
AUDIO_SOURCE = "Coquie_TTS" # @param ["Elevenlabs_TTS", "Coquie_TTS", "Google_TTS", "Edge_TTS", "RVC_TTS"]


COQUIE_LANGUAGES = "en" # @param ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
COQUIE_VOICES = "Badr Odhiambo" # @param ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Al


ELEVENLABS_LANGUAGES = "English" # @param ["English", "Chinese", "Spanish", "Hindi", "Portuguese", "French", "German", "Japanese", "Arabic", "Korean", "Indonesian", "Italian", "Dutch", "Turkish", "Polish", "Swedish", "Filipino", "Malay", "Russian", "Romanian", "Ukrainian", "Greek", "Czech", "Danish", "Finnish", "Bulgarian", "Croatian", "Slovak", "Tamil"]
ELEVENLABS_VOICE = "Bill"  # @param ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski"] {allow-input: true}
