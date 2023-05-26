import logging
import re
import time
from typing import List

from fastapi import HTTPException, status

from utils import CONFIG, DB
from utils.schemas import FilterRule, GetFiltersResponse, PostFilterResponse


async def create_filter_rule(vendor: str, organization: str, name: str, description: str | None, stop_words: List[str]):
    # checking if the rule with such name exists
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization].find_one({"rule_name": name})
    if result is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} already exists. Use PATCH method to update existing rule",
        )
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization]["archived"].find_one({"rule_name": name})
    if result is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} already exists and archived",
        )

    new_rule = {
        "rule_name": name,
        "description": description,
        "stop_words": stop_words,
        "timestamp": int(round(time.time())),
    }
    DB[CONFIG["mongo"]["filters"]][vendor][organization].insert_one(new_rule)
    logging.info(f"{vendor}.{organization}: Rule {name} created")
    return PostFilterResponse(name=name)


async def archive_filter_rule(vendor: str, organization: str, name: str):
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization].find_one({"rule_name": name})
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} does not exist",
        )
    result["timestamp"] = int(round(time.time()))  # updating timestamp
    DB[CONFIG["mongo"]["filters"]][vendor][organization]["archived"].insert_one(result)
    DB[CONFIG["mongo"]["filters"]][vendor][organization].delete_one({"_id": result["_id"]})
    logging.info(f"{vendor}.{organization}: Rule {name} archived")
    return PostFilterResponse(name=name)


async def update_filter_rule(vendor: str, organization: str, name: str, description: str | None, stop_words: List[str]):
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization].find_one({"rule_name": name})
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} does not exist",
        )
    DB[CONFIG["mongo"]["filters"]][vendor][organization].update_one(
        {"_id": result["_id"]},
        {"$set": {"description": description, "stop_words": stop_words, "timestamp": int(round(time.time()))}},
    )
    logging.info(f"{vendor}.{organization}: Rule {name} updated")
    return PostFilterResponse(name=name)


async def get_filters(vendor: str, organization: str):
    active_rules, archived_rules = [], []
    actives = DB[CONFIG["mongo"]["filters"]][vendor][organization].find({})
    for active_rule in actives:
        active_rules.append(
            FilterRule(
                name=active_rule["rule_name"],
                description=active_rule["description"],
                stop_words=active_rule["stop_words"],
            )
        )
    archived = DB[CONFIG["mongo"]["filters"]][vendor][organization]["archived"].find({})
    for archived_rule in archived:
        archived_rules.append(
            FilterRule(
                name=archived_rule["rule_name"],
                description=archived_rule["description"],
                stop_words=archived_rule["stop_words"],
            )
        )
    return GetFiltersResponse(active_rules=active_rules, archived_rules=archived_rules)


def check_filters(
    vendor: str,
    organization: str,
    query: str | None,
):
    if query is None:
        return
    clean_query = re.sub(
        r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
        " ",  # and replace it with a single space
        query.lower(),
        flags=re.VERBOSE,
    )
    word_set = set(clean_query.split(" "))
    word_set.remove("")

    all_rules = DB[CONFIG["mongo"]["filters"]][vendor][organization].find({})
    for rule in all_rules:
        for stopword in rule["stop_words"]:
            if stopword in word_set:
                logging.info(f"{vendor}.{organization}: query {query} failed check with rule {rule['rule_name']}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
                    detail=f"Request blocked by organization policy rule:\n{rule['rule_name']}\n{rule['description']}",
                )
