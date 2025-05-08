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
from huggingface_hub import login



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

class RAG:
    def __init__(self, pdf_path: str, model_name : str="all-mpnet-base-v2"):
        self.pdf_path = pdf_path
        self.model_name =  model_name 
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name)
        self.pages_and_texts = []
        self.embeddings = None
        self.llm_model = None 
        self.tokenizer = None 

    def download_pdf(self, url: str):
        if not os.path.exists(self.pdf_path):
            print(f"[INFO] PDF not found. Downloading from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(self.pdf_path, 'wb') as file:
                    file.write(response.content)
                print(f"[INFO] PDF downloading successfully")
            else:
                raise Exception(f"Failed to download PDF. status code {response.status_code}")
        else: 
            print(f"[INFO] PDF already exists at {self.pdf_path}")

    def read_and_process_pdf(self):
        doc = fitz.open(self.pdf_path)
        for page in tqdm(doc, desc="Reading PDF"):
            text = page.get_text().replace("\n ", " ").strip()
            self.pages_and_texts.append({
                "page_number": page.number,
                "text": text
            })

    def split_sentences(self):
        nlp = English()
        nlp.add_pipe("sentencizer")
        for page in self.pages_and_texts:
            doc = nlp(page["text"])
            page["sentences"] = [str(sent).strip() for sent in doc.sents]

    def chunk_text(self, chunk_size: int=10):
        for page in self.pages_and_texts:
            page["chunks"] = [page["sentences"][i:i+chunk_size] for i in range(0, len(page["sentences"]), chunk_size)]

    def embed_text_chunks(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.embedding_model.to(device)
        all_chunks = [" ".join(chunk) for page in self.pages_and_texts for  chunk in page.get("chunks", [])]
        self.embeddings = self.embedding_model.encode(all_chunks, batch_size=32, convert_to_tensor=True)

    def save_embeddings(self, file_path: str):
        data = []
        for page in self.pages_and_texts:
            for chunk in page.get("chunks", []):
                data.append({"text": " ".join(chunk)})
        pd.DataFrame(data).to_csv(file_path, index=False)

    def load_embeddings(self, file_path: str):
        df = pd.read_csv(file_path)
        self.pages_and_texts = df.to_dict(orient="records")

    def semantic_search(self, query: str, top_k: int=5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.dot_score(query_embedding, self.embeddings)[0]
        top_k_indices = torch.topk(scores, k=top_k).indices
        results = [self.pages_and_texts[i]["text"] for i in top_k_indices]
        return results 
    
    def load_llm(self, model_id: str,
                use_quantization_config: bool = False,
                 hf_token: str = None):
        
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        llm_int8_enable_fp32_cpu_offload=True)
        
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >=8):
            attn_implementation = "flash_attention_2" 
            print(f"[INFO] Using FlashAttention2 for model {model_id}.")
        else:
            attn_implementation = "sdpa" # scaled dot product attention
            print(f"[INFO] Using SDPA for model {model_id}.")


        self.tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                       token=hf_token)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.float16,
            quantization_config=quantization_config if use_quantization_config else None,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )

        # if not use_quantization_config:
        #     self.llm_model.to("cuda" if torch.cuda.is_available() else "cpu")

    def ask(self, query: str, temperature: float = 0.7,
            max_new_tokens: int = 256) -> str:
         
     
        context = self.semantic_search(query)
        context_text = "\n".join(context)
        # print(f"[INFO] context: {context_text}")

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
        
        base_prompt = base_prompt.format(context=context_text, query=query)

        # Create Template for instruction tuned Model
        dialogue_template = [
            {
                "role": "user",
                "content": base_prompt
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(conversation = dialogue_template,
                                            tokenize = False,
                                            add_genetration_prompt = True)
    
        # prompt = f"Based on the following context, answer the query:\n{context_text}\nQuery: {query}\nAnswer:"

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        output = self.llm_model.generate(**input_ids, 
                                         temperature=temperature,
                                          do_sample=True,
                                         max_new_tokens=max_new_tokens,
                                         )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.replace(prompt, "").strip()
     
if __name__ == "__main__":

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Hugging Face token not found! Please set HF_TOKEN in your .env file.")

    # 02. (Optional) Cache it locally so you can also use CLI-backed commands
    login(token=hf_token)

    # list of questions 

    # Nutrition-style questions generated with GPT4
    gpt4_questions = [
        "What are the macronutrients, and what roles do they play in the human body?",
        "How do vitamins and minerals differ in their roles and importance for health?",
        "Describe the process of digestion and absorption of nutrients in the human body.",
        "What role does fibre play in digestion? Name five fibre containing foods.",
        "Explain the concept of energy balance and its importance in weight management."
    ]

    # Manually created question list for nutrition
    manual_questions = [
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the RDI for protein per day?",
        "water soluble vitamins"
    ]

    query_list = gpt4_questions + manual_questions

            
    pdf_path = "human-nutrition-text.pdf"

    Rag_pipline = RAG(pdf_path=pdf_path)

    Rag_pipline.read_and_process_pdf()

    Rag_pipline.split_sentences()

    Rag_pipline.chunk_text(chunk_size=10)

    Rag_pipline.embed_text_chunks()

    Rag_pipline.save_embeddings("embeddings.csv")

    Rag_pipline.load_embeddings("embeddings.csv")

    Rag_pipline.load_llm(model_id="google/gemma-2-2b-it")

    query = random.choice(query_list)

    answer = Rag_pipline.ask(query=query, temperature=0.7, max_new_tokens=256)

    print(f"[INFO] query: {query}")
    print(f"[INFO] answer: {answer}")





