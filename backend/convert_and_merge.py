import json

closed_ids = set()
try:
    with open("meta-Rhode_Island.json", "r", encoding="utf-8") as f:
        for line in f:
            try:
                business = json.loads(line.strip())
                if "state" in business and business["state"] and "Permanently closed" in business["state"]:
                    closed_ids.add(business["gmap_id"])
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    print("meta-other.json not found. Skipping removal of closed businesses.")
    closed_ids = set()

keywords = ["food", "park", "amusement","lake","church"]

try:
    with open("meta-Rhode_Island.json", "r", encoding="utf-8") as f:
        for line in f:
            try:
                business = json.loads(line.strip())
                combined_text = " ".join(str(value) for value in business.values()).lower()
                if not any(keyword.lower() in combined_text for keyword in keywords):
                    closed_ids.add(business.get("gmap_id"))
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    print("meta-other.json not found. Skipping removal of closed businesses.")
    closed_ids = set()

input_path = "review-Rhode_Island.json"
output_path = "cleaned_reviews.json"

fields_to_remove = {"user_id", "name", "time", "pics"}

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            review = json.loads(line.strip())
            if review.get("gmap_id") in closed_ids:
                continue
            for field in fields_to_remove:
                review.pop(field, None)
            json.dump(review, outfile)
            outfile.write("\n")
        except json.JSONDecodeError:
            continue

with open("cleaned_reviews.json", "r", encoding="utf-8") as f:
    cleaned_reviews = [json.loads(line.strip()) for line in f if line.strip()]

with open("meta-Rhode_Island.json", "r", encoding="utf-8") as f:
    business_entries = [json.loads(line.strip()) for line in f if line.strip()]

business_dict = {b["gmap_id"]: b for b in business_entries if "gmap_id" in b}

merged_output = []
for review in cleaned_reviews:
    gmap_id = review.get("gmap_id")
    business = business_dict.get(gmap_id)
    if business:
        merged_entry = {
            "business": business,
            "review": review
        }
        merged_output.append(merged_entry)

with open("merged_reviews.json", "w", encoding="utf-8") as f:
    for item in merged_output:
        json.dump(item, f)
        f.write("\n")

input_json = "merged_reviews.json"
output_txt = "business_reviews.txt"

with open(input_json, "r", encoding="utf-8") as infile, open(output_txt, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            entry = json.loads(line.strip())
            biz = entry.get("business", {})
            rev = entry.get("review", {})

            text = f"Business Name: {biz.get('name')}\n"
            text += f"Category: {biz.get('category')}\n"
            text += f"Review: {rev.get('text')}\n"
            text += f"Rating: {rev.get('rating')}\n"
            text += f"Response: {rev.get('resp')}\n"
            text += f"Address: {biz.get('address')}\n"
            text += f"Rating: {biz.get('avg_rating')}\n"
            text += f"GMAP ID: {biz.get('gmap_id')}\n"
            text += f"Latitude: {biz.get('latitude')}\n"
            text += f"Longitude: {biz.get('longitude')}\n"
            text += f"Description: {biz.get('description')}\n"
            text += f"Average Rating: {biz.get('avg_rating')}\n"
            text += f"Price: {biz.get('price')}\n"
            text += f"Hours: {biz.get('hours')}\n"
            text += "---\n\n"
            outfile.write(text)
        except json.JSONDecodeError:
            continue
