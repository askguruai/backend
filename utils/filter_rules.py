from utils import CONFIG, DB
from typing import List
import logging
import time

from fastapi import HTTPException, status
from utils.schemas import PostFilterResponse


async def create_filter_rule(
    vendor: str,
    organization: str,
    name: str,
    description: str | None,
    stop_words: List[str]
):
    # checking if the rule with such name exists
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization].find_one(
        {"rule_name": name}
    )
    if result is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} already exists. Use UPDATE method to update existing rule",
        )
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization]["archived"].find_one(
        {"rule_name": name}
    )
    if result is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} already exists and archived",
        )

    new_rule = {
        "rule_name": name,
        "description": description,
        "stop_words": stop_words,
        "timestamp": int(round(time.time()))
    }
    DB[CONFIG["mongo"]["filters"]][vendor][organization].insert_one(new_rule)
    logging.info(f"Rule {name} created for {vendor}.{organization}")
    return PostFilterResponse(name=name)


async def archive_filter_rule(
    vendor: str,
    organization: str,
    name: str
):
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization].find_one(
        {"rule_name": name}
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} does not exist",
        )
    result["timestamp"] = int(round(time.time())) # updating timestamp
    DB[CONFIG["mongo"]["filters"]][vendor][organization]["archived"].insert_one(result)
    DB[CONFIG["mongo"]["filters"]][vendor][organization].delete_one({"_id": result["_id"]})
    logging.info(f"Rule {name} of {vendor}.{organization} archived")
    return PostFilterResponse(name=name)


async def update_filter_rule(
    vendor: str,
    organization: str,
    name: str,
    description: str | None,
    stop_words: List[str]
):
    result = DB[CONFIG["mongo"]["filters"]][vendor][organization].find_one(
        {"rule_name": name}
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # todo: more appropriate code
            detail=f"Rule with name {name} does not exist",
        )
    DB[CONFIG["mongo"]["filters"]][vendor][organization].update_one(
        {"_id": result["_id"]},
        {"$set": {"description": description, "stop_words": stop_words, "timestamp": int(round(time.time()))}}
    )
    logging.info(f"Rule {name} of {vendor}.{organization} updated")
    return PostFilterResponse(name=name)