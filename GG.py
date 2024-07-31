from g4f.client import Client
import json

client = Client()

topic = "Relationships and Connections"
goal = "help build meaningful relationships, foster connections, and spread love"

prompt_prefix = """You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding sentence that complements and enhances the visual. The sentence should be concise and powerful.
5. Ensure that the sequence of images and sentence builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:""".format(topic, goal)

sample_output = """
[
    { "image_description": "Description of the first image here.", "sentence": "sentence accompanying the first scene cut." },
    { "image_description": "Description of the second image here.", "sentence": "sentence accompanying the second scene cut." }
]"""

prompt_postinstruction = """By following these instructions, you will create an impactful {} short-form video.
Ensure your output is a valid JSON array. Do not include any sentence before or after the JSON array.
Output:""".format(topic)

prompt = prompt_prefix + sample_output + prompt_postinstruction

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],)

response_json = json.loads(response.choices[0].message.content)
image_prompts = [item['image_description'] for item in response_json]
sentences = [item['sentence'] for item in response_json]

print("Image Prompts:", image_prompts)
print("sentences:", sentences)
print("Number of sentences:", len(sentences))
