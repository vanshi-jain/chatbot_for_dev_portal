import os
import json
import time
from PIL import Image
import numpy as np
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize

from rouge import Rouge

from dotenv import load_dotenv
load_dotenv()

# Setup
llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()

# Load KB
def load_kb(folder="knowledge"):
    docs = []
    for f in os.listdir(folder):
        if f.endswith(".txt") or f.endswith(".json"):
            with open(os.path.join(folder, f), "r") as fp:
                content = fp.read()
            docs.append(Document(page_content=content, metadata={"source": f}))
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return FAISS.from_documents(splitter.split_documents(docs), embeddings)


rouge = Rouge()
smoother = SmoothingFunction().method1

def evaluate_correctness(expected, response):
    expected_words = set(expected.lower().split())
    response_words = set(response.lower().split())

    # 1. Half-word match rule
    common_word_match = len(response_words & expected_words) >= len(expected_words) / 2

    # 2. BLEU score
    bleu = sentence_bleu([expected.split()], response.split(), weights = [(1./2., 1./2.)],smoothing_function=smoother)
    bleu_match = bleu >= 0.7

    # 3. ROUGE score
    rouge_score = rouge.get_scores(response, expected)[0]["rouge-l"]["f"]
    rouge_match = rouge_score >= 0.5

    # 4. METEOR score
    meteor = single_meteor_score(word_tokenize(expected), word_tokenize(response))
    meteor_match = meteor >= 0.5

    # Final correctness (True if any 2 or more conditions are met)
    score_flags = [common_word_match, bleu_match, rouge_match, meteor_match]
    final_correct = sum(score_flags) >= 1

    return {
        "correct": final_correct,
        "common_word_match": common_word_match,
        "bleu": bleu,
        "rouge_l": rouge_score,
        "meteor": meteor
    }

# Evaluate
def evaluate(test_json, image_dir="flowcharts", ocr_dir="processed", know_dir="knowledge"):
    vectorstore = load_kb()
    retriever = vectorstore.as_retriever()
    
    with open(test_json) as f:
        tests = json.load(f)

    correct, total = 0, 0
    timings = []
    results = []
    
    for sample in tests:
        # image_path = os.path.join(image_dir, sample["image"])
        # processed_ocr = os.path.join(ocr_dir, sample["image"].replace('png', 'txt'))
        knowledge_doc = os.path.join(know_dir, sample["image"].replace('png', 'json'))
        # print(knowledge_doc)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )
        for q in sample["queries"]:
            start = time.time()
            response = chain.run({
                "flow_text": knowledge_doc,
                "query": q['question']
                })
            end = time.time()
            # print(response)     

            latency = round(end - start, 2)
            timings.append(latency)

            expected = q["expected"]
            res_score = evaluate_correctness(expected, response)
            # bleu = sentence_bleu([expected.split()], response.split(), smoothing_function=SmoothingFunction().method1)
            # meteor = single_meteor_score(expected, response)

            # rouge = Rouge()
            # rouge_score = rouge.get_scores(response, refs=expected)[0]["rouge-l"]["f"]


            # is_correct = expected.lower() in response.lower()
            # correct += int(is_correct)
            correct += int(res_score['correct'])
            total += 1

            results.append({
                "image": sample["image"],
                "query": q["question"],
                "expected": expected,
                "response": response,
                "latency": latency,
            } | res_score
            )

            # if q["expected"].lower() in response.lower():
            #     correct += 1
            # total += 1
            # print(f"Q: {q['question']} → ✅ {'YES' if q['expected'].lower() in response.lower() else 'NO'} | {latency}s")

    acc = correct / total * 100
    avg_latency = sum(timings) / len(timings)
    print(f"\nFinal Accuracy: {acc:.2f}%")
    print(f"Average Latency: {avg_latency:.2f} seconds")

    summary = {
        "total": total,
        "correct": correct,
        "accuracy_percent": acc,
        "average_latency": avg_latency
    }

    output = {
        "summary": summary,
        "details": results
    }

    with open("./res.json", "w") as f:
        json.dump(output, f, indent=2)

    return acc, avg_latency


evaluate("./test.json")