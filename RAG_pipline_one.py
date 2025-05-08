# importing necessary libraries 
import torch 
import os 
import requests 
import fitz 
from tqdm.auto import tqdm 
import random 
import pandas as pd 
import re
from spacy.lang.en import English 
from sentence_transformers import SentenceTransformer
import numpy as np
from time import perf_counter as timer 
from sentence_transformers import util, SentenceTransformer
import textwrap 
import matplotlib.pyplot as plt 
from dotenv import load_dotenv
from huggingface_hub import login 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import is_flash_attn_2_available



# path to pdf and if it exists else download it

pdf_path = "human-nutrition-text.pdf"

if not os.path.exists(pdf_path):
    print(f"[INFO] file doesn't exists, downloading...")

    # Enter the url of the pdf 
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

    # the local file name to save the downloaded file 
    filename = pdf_path

    # send a get request to the url
    response = requests.get(url)

    # check if the request was succesfull

    if response.satus_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"[INFO] file downloaded successfully and saved as {filename}")
    else:
        print(f"[ERROR] failed to download the file, status code: {response.status_code}")
else:
    print(f"[INFO] file already exists at {pdf_path}")

# Text formatter 

def text_formatter(text:str) -> str:
    """ Perform minor formatting"""
    cleaned_text = text.replace("\n", " ").strip()
    
    # any remaining cleaning codes can go here
    return cleaned_text
def open_and_read_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number - 41, # since the page 1 starts from page 42 on the pdf we are using 
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")) ,
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text)/4,
            "text": text
        })
    return pages_and_texts
pages_and_texts = open_and_read_pdf(pdf_path)



##################################################
# text preprocessing further 
nlp = English()
nlp.add_pipe("sentencizer")

# making sure all the sentences are strings 

for item in tqdm(pages_and_texts):

    item["sentences"] = list(nlp(item["text"]).sents)
    
    #making sure all the sentences are strings 
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]

    # count the sentences 
    item["pages_sentence_count_spacy"] = len(item["sentences"])


################################################

# Chunking the sentences into smaller chunks
num_sentence_chunk_size = 10 # number of sentences in each chunk

def split_list(input_list: list[str], slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
    """Splits a list into smaller chunks of specified size."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# loop through pages and text and split sentences into chunks 

for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list = item["sentences"],
                                         slice_size = num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])


###############################################################

# splitting each chunks into its own item 

pages_and_chunks = []

for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(sentence_chunk).replace(" ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'\1', joined_sentence_chunk)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # get some stats on our chunk 
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token ~=4 chars

        pages_and_chunks.append(chunk_dict)

##########################################
df = pd.DataFrame(pages_and_chunks)
# filtering the text data more 
min_token_length = 30 
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(f"Chunk teoken count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}")

# filter dataframe for rows under 30 token 

pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
# pages_and_chunks_over_min_token_len

######################################################################
# Embedding our text chunks 

embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2",
                                     device="cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model.to(device) # setting the mode into cuda if availabel 

for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

# embeddingd all texts in batches 
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32,
                                               convert_to_tensor=True)

# Saving embeddings to file 
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path)

# Importing saved file and view 
text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)

# Import texts and embeddings 
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# Convert embeddings column back to np.array
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert embeddings into torch tensors 
embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist(), axis=0), dtype=torch.float32).to(device)

# Convert texts and embeddings df 
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# text_chunks_and_embeddings_df 
#####################################################################################
# Create embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                     device = device)

# Define the query 
# query = "macronutrinets functions"

# # 2. Embed the query
# query_embedding = embedding_model.encode(query, convert_to_tensor=True).to("cuda")

# # 3. Get similaarity scores with the dot product(use cosine similaryt search)
# # To use the dot product the dtype and shape must much!

# start_time = timer()
# dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
# end_time = timer()

# print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds")

# # 4. Get the top k e.g top five
# top_results_dot_product = torch.topk(dot_scores, k=5)
# top_results_dot_product
#####################################################################################################################

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    return wrapped_text

###############################################################################

def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm distance
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    # Get the cosine dot product
    cosine_dot_product = dot_product / (norm_vector1 * norm_vector2)
    
    return cosine_dot_product


# Semantic search pipline 

def retrieve_relevant_resources(query: str, embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    
    """Embedd a query with embeddings model and return
      top k scores and indices from embeddings.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Get the dot product on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds")

    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)

    return scores, indices 

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """ Finds relevant passages given a query 
     and prints them out along with their scores.
       """
    scores, indices = retrieve_relevant_resources(query=query,
                                                 embeddings=embeddings,
                                                 n_resources_to_return=n_resources_to_return)
    # Loop through zipped scores and indices 
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text")
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        print(f"page number: {pages_and_chunks[index]["page_number"]}")
        print("\n")


query = "food high in fiber"
retrieve_relevant_resources(query=query,embeddings=embeddings)

print_top_results_and_scores(query=query, embeddings=embeddings)

######################################################################
# Getting or instantiating the LLM

gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
print(f"Available GPU memory: {gpu_memory_gb} GB")

if gpu_memory_gb < 5.1:
    print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization")
elif gpu_memory_gb < 8.1:
    print(f"GPU memory: {gpu_memory_gb} | Recommende model: Gemma 2b in 4-bit precision")
    use_quantization_config = True
    model_id = "google/gemma-2-2b-it"
elif gpu_memory_gb < 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommende model: Gemma 2b in float16 or 4-bit precision")
    use_quantization_config = False
    model_id = "google/gemma-2-2b-it"
elif gpu_memory_gb > 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommende model: Gemma 7b in float16 or 4-bit precision")
    use_quantization_config = False
    model_id = "google/gemma-7b-it"

print(f"use_quantization_config set to: {use_quantization_config}")
print(f"model_id set to: {model_id}")
    

# 1. Loading the environment variable 
load_dotenv()

# 2. Grab token 
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Hugging Face toekn not found! Please set HF_TOKEN in your .env file.")

login(token=hf_token)

# 3. Create a quantization config 
quantization_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_compute_dtype=torch.bfloat16,
     llm_int8_enable_fp32_cpu_offload=True,
 )

# 4. Check if flash attention is available

if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >=8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

# 5. pick a model 
model_id = "google/gemma-2-2b-it"
model_id = model_id

# 6. Instantiate the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id,
    token=hf_token,
)

# 7 Instantiate the mode 
llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    device_map="auto",
    token=hf_token,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    attn_implementation=attn_implementation,
)

if not use_quantization_config:
    llm_model.to(device)

##########################################################
# Generating text with our LLM

# Nutrition-style questions generated with GPT4
gpt4_questions = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "Describe the process of digestion and absorption of nutrients in the human body.",
    "What role does fibre play in digestion? Name five fibre containing foods.",
    "Explain the concept of energy balance and its importance in weight management."
]

# Manually created question list
manual_questions = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins"
]

query_list = gpt4_questions + manual_questions
# query_list

# Augmenting prompt with context 

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "_ " + "\n ".join([item["sentence_chunk"] for item in context_items])

    base_prompt = """Based on the following context items, please answer the query.
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible.
        Use the following examples as reference for the ideal answer style.
        \nExample 1:
        Query: What are the fat-soluble vitamins?
        Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
        \nExample 2:
        Query: What are the causes of type 2 diabetes?
        Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
        \nExample 3:
        Query: What is the importance of hydration for physical performance?
        Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
        \nNow use the following context items to answer the user query:
        {context}
        \nRelevant passages: <extract relevant passages from the context here>
        User query: {query}
        Answer:"""
    
    base_prompt = base_prompt.format(context=context, query=query)

    # Create Template for instruction tuned Model 
    dialogue_template = [
        {
        "role": "user",
        "content": base_prompt
      }
    ]

    prompt = tokenizer.apply_chat_template(
        conversation = dialogue_template,
        tokenize = False,
        add_generation_prompt = True 
    )

    return prompt

query = random.choice(query_list)
print(f"Query: {query}")

# Get the releveant resources 

scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)

# Creating a list of context items 
context_items = [pages_and_chunks[i] for i in indices]

prompt = prompt_formatter(query=query, context_items=context_items)

# print(prompt)
#############################
# Generation 
input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# Sentence on output of tokens 
outputs = llm_model.generate(**input_ids,
                            temperature=0.7,
                            do_sample=True,
                            max_new_tokens=500)

output_text = tokenizer.decode(outputs[0])

print(f"Query: {query}")

print(f"Rag Answer: \n{output_text.replace(prompt, '')}")


# Functionizing the LLM response 

def ask(query: str, temperature: float = 0.7, max_new_tokens: int=256,
        format_answer_text=True, return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and genarte 
    an answer to the query based on the relevant resources."""
    
    # Get just the scores and indices of the top related 
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings
    )

    # Create a list of context items 
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context items 
    for i, items in enumerate(context_items):
        items["score"] = scores[i].cpu()

    # Augmenting the prompt with context
    prompt = prompt_formatter(query=query, context_items = context_items)

    # Tokenize the prompt 
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)


    # Generate the aanswer 
    output = llm_model.generate(**input_ids, 
                                 temperature=0.7,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    # Decode the tokens 
    output_text = tokenizer.decode(output[0])

    # Format the answer 
    if format_answer_text:
        # Replace prompt and special 
        output_text = output_text.replace(prompt, '').replace("<bos>", "").replace("<eos>", "").replace("<end_of_turn>", "")

    # Only return the answer 
    if return_answer_only:
        return output_text
    
    return output_text, context_items
query = random.choice(query_list)
print(f"Query: {query}")
ask(query=query, temperature=0.2, return_answer_only=True)
