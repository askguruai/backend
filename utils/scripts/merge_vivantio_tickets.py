import json
import os

FOLDER = "./data/vivantio_tickets_new"

OUTPUT = "./data/vivantio_tickets.json"

from loguru import logger

# each file is a list of dicts
# each dict is a ticket
# we want to extract only those tickets
# that have a field "summary"
#
# tickets might duplicate
#
# load files via json


def load_tickets(folder):
    tickets = []
    processed = set()
    for file in os.listdir(folder):
        cnt = 0
        with open(os.path.join(folder, file), "r") as f:
            for ticket in json.load(f):
                # print(ticket["hdcDescription"])
                if (
                    ticket["hdccallername"] != "Loggly"
                    and "Internal Only" not in (ticket["hdcatLineage"] or "")
                    and "Alert" not in ticket["hdccallername"]
                    and "Rackspace" not in ticket["hdccallername"]
                    and ticket["hdcDescription"]
                    and "Vivantio Monitoring Tools" not in ticket["hdcDescription"]
                    and "summary" in ticket
                    and ticket["idhdcall"] not in processed
                ):
                    tickets.append(ticket)
                    processed.add(ticket["idhdcall"])
                    cnt += 1
        logger.info(f"Loaded {cnt} tickets from {file}")

    # write to output
    with open(OUTPUT, "w") as f:
        json.dump(tickets, f, indent=4)

    return tickets


logger.info(len(load_tickets(FOLDER)))
