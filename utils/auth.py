import asyncio
import os
from typing import List

import requests
from fastapi import Body, Depends, HTTPException, Path, Query, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger
from pydantic import Field

from utils import CLIENT_SESSION_WRAPPER
from utils.schemas import (
    ApiVersion,
    AuthenticatedRequest,
    LivechatLoginRequest,
    VendorCollectionRequest,
    VendorCollectionTokenRequest,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/collections/token")


def create_access_token(data: dict):
    return jwt.encode(data, os.environ["JWT_SECRET_KEY"], algorithm=os.environ["JWT_ALGORITHM"])


async def get_organization_token(
    api_version: ApiVersion,
    vendor: str = Body(description="Vendor name", example="livechat"),
    organization: str = Body(description="Organization within vendor", example="f1ac8408-27b2-465e-89c6-b8708bfc262c"),
    password: str = Body(..., description="This is for staff use"),
    security_groups: List[int] = Body(
        None, description="Security groups associated with token. Leave blank for full access"
    ),
):
    if password == os.environ["AUTH_COLLECTION_PASSWORD_ASKGURUPUBLIC"] and vendor != "askgurupublic":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You are only allowed to access organizations from askgurupublic vendor",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if password == os.environ["AUTH_COLLECTION_PASSWORD_TADA"] and vendor != "tada":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You are only allowed to access organizations from tada vendor",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if password not in [
        os.environ["AUTH_COLLECTION_PASSWORD"],
        os.environ["AUTH_COLLECTION_PASSWORD_TADA"],
        os.environ["AUTH_COLLECTION_PASSWORD_ASKGURUPUBLIC"],
    ]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    security_groups = [] if security_groups is None else security_groups
    access_token = create_access_token(
        {"vendor": vendor, "organization": organization, "security_groups": tuple(security_groups)}
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def get_livechat_token(api_version: ApiVersion, livechat_token: str = Body(...)):
    headers = {
        "Authorization": f"Bearer {livechat_token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    async def fetch(method, url, json=None):
        async with CLIENT_SESSION_WRAPPER.general_session.request(method, url, headers=headers, json=json) as response:
            return response.status, await response.json()

    tasks = [
        fetch("GET", "https://accounts.livechatinc.com/v2/info"),
        fetch("GET", "https://accounts.livechatinc.com/v2/accounts/me"),
        fetch(
            "POST",
            "https://api.livechatinc.com/v3.5/configuration/action/list_groups",
            {"fields": ["agent_priorities"]},
        ),
    ]
    responses = await asyncio.gather(*tasks)

    for response_status, _ in responses:
        if response_status != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    organization = responses[0][1]["organization_id"]
    agent_identifier = responses[1][1]["email"]
    security_groups = tuple(
        group["id"]
        for group in responses[2][1]
        if group["agent_priorities"] is not None and agent_identifier in group["agent_priorities"]
    )

    logger.info(f"Livechat user {agent_identifier} is in security groups {security_groups}")

    access_token = create_access_token(
        data={"vendor": "livechat", "organization": organization, "security_groups": security_groups}
    )
    return {"access_token": access_token, "token_type": "bearer"}


def decode_token(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials/token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            os.environ["JWT_SECRET_KEY"],
            algorithms=[os.environ["JWT_ALGORITHM"]],
        )

    except JWTError:
        raise credentials_exception
    return payload
