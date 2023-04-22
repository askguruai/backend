from argparse import ArgumentParser
import os, os.path as osp
import json
import openai
from tqdm import tqdm

PROMPT = "Following is an email conversation with support. Extract the problem description from the conversation" \
         "as verbose as possible, cite where necessary. If the solution is present, extract it in the same way. " \
         "Present answer strictly as a json object with two fields: 'problem' and 'solution'. If there is no solution" \
         "present, put null"

to_prompt = lambda text: f"{PROMPT}\n{text}"


def retrieve_summary(ticket: dict):
    text_data = ticket["hdcDescription"]
    # meta = {
    #     "lineage": ticket["hdcatLineage"],
    #     "ticket_id": ticket["idhdcall"],
    #     "title": ticket["hdctitle"]
    # }
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": to_prompt(text_data[:3000])},
        ],
        temperature=0.7,
        max_tokens=500,
        presence_penalty=0.6,
    )["choices"][0]["message"]["content"].lstrip()
    print(f"Returned answer: {answer}")
    try:
        summary = json.loads(answer)
        if "problem" not in summary or "solution" not in summary:
            return {"raw": answer}
        return summary
    except json.decoder.JSONDecodeError as e:
        return {"raw": answer}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="json ticket dump")
    args = parser.parse_args()

    with open(args.source, "rt") as f:
        data = json.load(f)

    for ticket in tqdm(data[:50]):
        if "summary" in ticket:
            continue
        summary = retrieve_summary(ticket)
        ticket["summary"] = summary
    with open(args.source, "wt") as f:
        json.dump(data, f)
