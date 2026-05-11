# Hugging-Face-Transformers-Mini-Tasks

📦 HUGGING FACE TRANSFORMERS – COMPLETE MINI PROJECTS
======================================================

A hands-on collection of NLP and generative AI tasks using pretrained models from Hugging Face.
🧩 Includes: Spam detection 📝 summarization 🌍 translation ❓ QA 🖼️ text-to-image 🎬 text-to-video.

----------------------------------------------------------------------------------------------------

🔧 SETUP
--------

Install the required libraries (one-time):

pip install torch transformers diffusers accelerate imageio[ffmpeg] opencv-python Pillow matplotlib ipython

💡 Note: For text-to-video, a GPU with at least 8 GB VRAM is strongly recommended.

----------------------------------------------------------------------------------------------------

1. 🛡️ SPAM DETECTION
--------------------

from transformers import pipeline

spam_classifier = pipeline("text-classification", model="philschmid/distilbert-base-multilingual-cased")

texts = [
    "Congratulations! You've won a 500 INR Amazon gift card. Click here to claim now.",
    "Hi Amit, let's have a meeting tomorrow at 12 PM.",
    "URGENT: Your gmail account has been compromised. Click here to secure it."
]

results = spam_classifier(texts)
label_mapping = {'negative': 'SPAM', 'neutral': 'NOT SPAM', 'positive': 'NOT SPAM'}

for result in results:
    label = label_mapping[result['label']]
    print(f"Label: {label}, Confidence: {result['score']:.4f}")

----------------------------------------------------------------------------------------------------

2. 📝 TEXT SUMMARIZATION (BART)
-------------------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """
Virat Kohli is an Indian international cricketer known for his exceptional batting skills. 
He has played for the Indian national team in all formats and is a former captain. 
Kohli is referred to as "King Kohli" and "Chase Master" due to his remarkable ability to chase down targets.
"""

inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs, max_length=50, min_length=25, length_penalty=2.0)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)

----------------------------------------------------------------------------------------------------

3. 🌍 TRANSLATION (ENGLISH → SPANISH) WITH T5
---------------------------------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "Translate English to Spanish: My name is Amit Diwan, and I love cricket."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50)
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translated:", translated)

----------------------------------------------------------------------------------------------------

4. ❓ QUESTION ANSWERING (BERT ON SQUAD)
----------------------------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = """
Emma Johnson is a marine biologist based in Sydney, Australia. She works at the Australian Institute of Marine Science. 
Her research focuses on coral reef restoration. In her spare time, she enjoys scuba diving and photography.
"""

question = "Where is Emma Johnson based?"
inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
outputs = model(**inputs)

start_idx = outputs.start_logits.argmax()
end_idx = outputs.end_logits.argmax()
answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
print("Answer:", answer)

----------------------------------------------------------------------------------------------------

5. 🖼️ TEXT-TO-IMAGE (STABLE DIFFUSION)
---------------------------------------

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

prompt = "Flying cars soar over a futuristic cityscape at sunset."
with torch.autocast(device):
    image = pipe(prompt).images[0]

image.save("generated_image.png")
print("Image saved as generated_image.png")

----------------------------------------------------------------------------------------------------

6. 🎬 TEXT-TO-VIDEO (MODELSCOPE)
--------------------------------

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()          # saves GPU memory

prompt = "A panda playing a guitar on a cloud, animated style"
video_frames = pipe(prompt, num_frames=24, num_inference_steps=25).frames[0]

export_to_video(video_frames, "generated_video.mp4", fps=8)
print("Video saved as generated_video.mp4")

# To view in a Jupyter notebook / Colab:
from IPython.display import Video
Video("generated_video.mp4", width=640)

----------------------------------------------------------------------------------------------------

📄 REQUIREMENTS (requirements.txt)
----------------------------------

torch>=2.0.0
transformers>=4.30.0
diffusers>=0.24.0
accelerate>=0.20.0
imageio[ffmpeg]>=2.31.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
ipython>=8.12.0

----------------------------------------------------------------------------------------------------

🚀 HOW TO RUN EVERYTHING
------------------------

1️⃣ Copy the code blocks above into separate .py files (or one big script).
2️⃣ Run each script individually.
3️⃣ Models are downloaded automatically on first use (internet connection required).
4️⃣ For text-to-video, be patient – generation takes 1-2 minutes on a good GPU.

----------------------------------------------------------------------------------------------------

📜 LICENSE
----------
MIT – free to use, modify, and share.
