# evaluation.py
import pandas as pd
from recommendation import run_recommendation

def run_evaluation():
    input_csv = r"C:\Users\HP\OneDrive\Desktop\myca_eval_dataset.csv"
    output_csv = r"C:\Users\HP\OneDrive\Desktop\myca_new_dataset.csv"

    df = pd.read_csv(input_csv)

    results = []
    messages = []

    for idx, row in df.iterrows():
        text = row["context"] + "\n" + row["conversation"]

        data, final_msg = run_recommendation(text)

        results.append(data.get("recommendations"))
        messages.append(final_msg)

    df["model_recommendations"] = results
    df["model_message"] = messages

    df.to_csv(output_csv, index=False)
    print("Saved results to:", output_csv)


if __name__ == "__main__":
    run_evaluation()
