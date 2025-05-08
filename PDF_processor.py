import os
import requests
import fitz
import re
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class PDFTextProcessor:
    def __init__(self, pdf_path: str, model_name: str = "all-mpnet-base-v2"):
        self.pdf_path = pdf_path
        self.model_name = model_name
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
                print("[INFO] PDF downloaded successfully.")
            else:
                raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
        else:
            print(f"[INFO] PDF already exists at {self.pdf_path}.")

    def read_and_process_pdf(self):
        doc = fitz.open(self.pdf_path)
        for page in tqdm(doc, desc="Reading PDF"):
            text = page.get_text().replace("\n", " ").strip()
            self.pages_and_texts.append({"page_number": page.number, "text": text})

    def split_sentences(self):
        nlp = English()
        nlp.add_pipe("sentencizer")
        for page in self.pages_and_texts:
            doc = nlp(page["text"])
            page["sentences"] = [str(sent).strip() for sent in doc.sents]

    def chunk_text(self, chunk_size: int = 10):
        for page in self.pages_and_texts:
            page["chunks"] = [page["sentences"][i:i+chunk_size] for i in range(0, len(page["sentences"]), chunk_size)]

    def embed_text_chunks(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.embedding_model.to(device)
        all_chunks = [" ".join(chunk) for page in self.pages_and_texts for chunk in page.get("chunks", [])]
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

    def semantic_search(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.dot_score(query_embedding, self.embeddings)[0]
        top_k_indices = torch.topk(scores, k=top_k).indices
        results = [self.pages_and_texts[i]["text"] for i in top_k_indices]
        return results

    def load_llm(self, model_id: str, use_quantization: bool = False, hf_token: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if not use_quantization else torch.bfloat16,
            load_in_4bit=use_quantization
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def ask(self, query: str, temperature: float = 0.7, max_new_tokens: int = 256) -> str:
        context = self.semantic_search(query)
        context_text = "\n".join(context)
        prompt = f"Based on the following context, answer the query:\n{context_text}\nQuery: {query}\nAnswer:"

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        output = self.llm_model.generate(**input_ids, max_new_tokens=max_new_tokens, temperature=temperature)

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.replace(prompt, "").strip()


pdf_path = "example.pdf"  # Change this to your PDF file path or URL
pdf_processor = PDFTextProcessor(pdf_path)

# Download the PDF (if it's a URL)
# pdf_processor.download_pdf("https://example.com/path/to/your/pdf.pdf")

# Read and process the PDF
pdf_processor.read_and_process_pdf()
pdf_processor.split_sentences()
pdf_processor.chunk_text(chunk_size=10)

# Embed the text chunks
pdf_processor.embed_text_chunks()

# Save the embeddings (optional)
pdf_processor.save_embeddings("embeddings.csv")

# Load the embeddings (if you have previously saved them)
# pdf_processor.load_embeddings("embeddings.csv")

# Load a Language Model (LLM) - Replace with your preferred model ID
pdf_processor.load_llm(model_id="openlm-research/open_llama_3b_v2")

# Ask a question using the LLM with context from the PDF
query = "What is the main topic discussed in the document?"
answer = pdf_processor.ask(query)
print("Answer:", answer)