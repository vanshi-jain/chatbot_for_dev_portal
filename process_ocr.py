import json
from typing import List, Dict
import numpy as np
from collections import defaultdict
import os
from utils.actors_list import ACTORS


def get_center(bbox: List[List[int]]):
    """Get the center of the polygon bbox."""
    pts = np.array(bbox)
    return np.mean(pts[:, 0]), np.mean(pts[:, 1])

def load_and_sort_by_position(data: Dict) -> List :
    texts = data["rec_texts"]
    bboxes = data["dt_polys"]

    merged = []
    for text, bbox in zip(texts, bboxes):
        if len(text.strip()) <= 1:
            continue  # skip noise
        cx, cy = get_center(bbox)
        merged.append({"text": text.strip(), "bbox": bbox, "cx": cx, "cy": cy})

    # Sort top-to-bottom, then left-to-right (this should be optimized based on arrows)
    merged = sorted(merged, key=lambda x: (x["cy"], x["cx"]))
    return merged

def group_lines(sorted_blocks, y_threshold=20):
    """Group text fragments into lines based on vertical proximity."""
    grouped = []
    current_line = []
    last_y = None

    for item in sorted_blocks:
        if last_y is None or abs(item["cy"] - last_y) < y_threshold:
            current_line.append(item["text"])
        else:
            grouped.append(" ".join(current_line))
            current_line = [item["text"]]
        last_y = item["cy"]

    if current_line:
        grouped.append(" ".join(current_line))

    return grouped

def group_by_actor(blocks, actor_threshold=35):
    actor_map = {}
    text_lines = []

    for block in blocks:
        txt = block["text"].upper()
        if txt in ACTORS:
            actor_map[txt] = block["cy"]
        else:
            text_lines.append(block)

    actor_assignments = defaultdict(list)
    for line in text_lines:
        assigned = False
        for actor, y_ref in actor_map.items():
            if abs(line["cy"] - y_ref) < actor_threshold:
                actor_assignments[actor.title()].append(line["text"])
                assigned = True
                break
        if not assigned:
            actor_assignments["System"].append(line["text"])  # fallback

    return actor_assignments

def format_grouped_steps(actor_assignments):
    structured = []
    step = 1
    for actor, texts in actor_assignments.items():
        if not texts:
            continue
        structured.append({
            "step": step,
            "actor": actor,
            "event": " ".join(texts)
        })
        step += 1
    return structured

def extract_structure(lines):
    steps = []
    current_actor = "System"
    step_num = 1

    for line in lines:
        actor_found = [a for a in ACTORS if a in line.upper()]
        if actor_found:
            current_actor = actor_found[0].title()
            continue

        event = line

        if line.strip().upper() in ["YES", "NO"]:
            continue  # skip flow-only links

        if "Is" in line or "?" in line:
            step_type = "Decision"
        elif "Cancel" in line or "Decline" in line:
            step_type = "Flow"
        else:
            step_type = "Process"

        steps.append({
            "step": step_num,
            "event": event,
            "actor": current_actor,
            "type": step_type
        })
        step_num += 1

    return steps
    
def group_by_keywords(parsed_data):
    raw_texts = parsed_data["rec_texts"]

    # remove standalone noise like "E" or single letters
    texts = [t.strip() for t in raw_texts if len(t.strip()) > 1]

    # Naive grouping based on keywords (can be made smarter)
    grouped_steps = []
    current_step = []

    decision_words = {"YES", "NO"}

    for word in texts:
        if word in ACTORS or word in decision_words:
            if current_step:
                grouped_steps.append(" ".join(current_step))
                current_step = []
            current_step.append(word)
        else:
            current_step.append(word)

    if current_step:
        grouped_steps.append(" ".join(current_step))
    return grouped_steps

if __name__ == "__main__":
    image_list = [ f for f in os.listdir("./data")
                   if not f.startswith(".")  
                ]
    out_path = "./processed"

    for x in image_list:
        json_path = f"./data/{x}/OCR_output/{x.replace('input', 'image')}.json"

        with open(json_path) as f:
            json_data = json.load(f)

        items = load_and_sort_by_position(json_data)
        formattype = "json"

        # Option 1: group by actor and format json structure
        structured_flow=format_grouped_steps(group_by_actor(items))

        # Option 2: group by lines and format json structure by actor 
        structured_flow = extract_structure(group_lines(items))

        # Option 3: group by keywords
        formattype = "txt"
        structured_flow = group_by_keywords(json_data)

        if formattype == "txt":
            with open(f"{out_path}/{x}.{formattype}", "w") as out:
                out.write("\n".join(f"{i+1}. {line}" for i, line in enumerate(structured_flow)))
        else:
            with open(f"{out_path}/{x}.{formattype}", "w") as out:
                json.dump(structured_flow, out, indent=2)   


