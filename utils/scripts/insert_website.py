# pip install "typer[all]"

from enum import Enum
from urllib.parse import urlencode

import requests
import typer
from loguru import logger
from typing_extensions import Annotated

VENDOR = "askgurupublic"
PASSWORD = "qy3vKVDVUtzCYDIZbYFozXlBp"


class ApiVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"


class BackendUrl(str, Enum):
    prod = "https://api.askguru.ai"
    dev1 = "https://api-dev1.askguru.ai"
    dev2 = "https://api-dev2.askguru.ai"


def insert_website(
    link: Annotated[str, typer.Argument(help="Link to parse. Should contain http/https in it.")],
    organization: str = None,
    query: str = None,
    summarize: bool = False,
    api_version: ApiVersion = ApiVersion.v2.value,
    backend_url: BackendUrl = BackendUrl.prod.value,
):
    if not link.startswith("http"):
        raise ValueError("Link should start with http/https")

    api_url = f"{backend_url}/{api_version}"

    # extract website name before dot
    website = link.replace("www.", "").split("//")[1].split(".")[0] if organization is None else organization

    if query is not None:
        params = urlencode({"org": website, "query": query})
        full_link = f"https://app.askguru.ai/?{params}"
        print(f"{full_link}")
        return

    logger.info(f"Inserting website {link} into collection '{VENDOR}_{website}_website' via {api_url}")
    logger.info(f"After the insertion, collection will be available at https://app.askguru.ai/?org={website}")

    token = requests.post(
        f"{api_url}/collections/token",
        json={"vendor": VENDOR, "organization": website, "password": PASSWORD},
    ).json()["access_token"]
    logger.info("Received token")

    logger.info("Inserting...")
    n_chunks = requests.post(
        f"{api_url}/collections/website",
        json={"links": [link], "summarize": summarize},
        headers={"Authorization": f"Bearer {token}"},
    ).json()["n_chunks"]
    logger.info(f"Inserting finished in {n_chunks} chunks!")


if __name__ == "__main__":
    typer.run(insert_website)
