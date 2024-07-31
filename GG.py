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

  image_prompts, sentences = fetch_imagedescription_and_script(prompt)

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
