from argparse import ArgumentParser
import os, os.path as osp
import json
import openai
from tqdm import tqdm
from openai.error import OpenAIError
import time
import os
# os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

PROMPT = "Following is an email conversation with support. Extract the problem description from the conversation" \
         "as verbose as possible, cite where necessary. If the solution is present, extract it in the same way. " \
         "Present answer strictly as a json object with two fields: 'problem' and 'solution'. If there is no solution" \
         "present, put null"

to_prompt = lambda text: f"{PROMPT}\n{text}"
def retrieve_summary(ticket: dict):
    text_data = ticket["hdcDescription"]
    if text_data is None:
        return {"raw": ""}
    # meta = {
    #     "lineage": ticket["hdcatLineage"],
    #     "ticket_id": ticket["idhdcall"],
    #     "title": ticket["hdctitle"]
    # }
    done = False
    while not done:
        try:
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": to_prompt(text_data[:3000])},
                ],
                temperature=0.7,
                max_tokens=500,
                presence_penalty=0.0,
            )["choices"][0]["message"]["content"].lstrip()
            done = True
        except OpenAIError as e:
            print(f"OpenAIError: {e._message}")
            time.sleep(3)
    try:
        summary = json.loads(answer)
        if summary is None:
            return {"raw": answer}
        if "problem" not in summary or "solution" not in summary:
            return {"raw": answer}
        return summary
    except json.decoder.JSONDecodeError as e:
        return {"raw": answer}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="json ticket dump")
    parser.add_argument("-o", "--out", type=str, help="json new ticket dump")
    args = parser.parse_args()

    with open(args.source, "rt") as f:
        data = json.load(f)

    for i, ticket in tqdm(enumerate(data[20000:])):
        if "summary" in ticket:
            continue
        summary = retrieve_summary(ticket)
        ticket["summary"] = summary
        if (i + 1) % 100 == 0:
            with open(args.out, "wt") as f:
                json.dump(data, f, indent=4)
