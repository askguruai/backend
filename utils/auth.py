import os

import requests
from fastapi import Body, Depends, HTTPException, Path, Query, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import Field

from utils.schemas import (
    ApiVersion,
    AuthenticatedRequest,
    LivechatLoginRequest,
    UploadChatsRequest,
    VendorCollectionRequest,
    VendorCollectionTokenRequest,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict):
    return jwt.encode(data, os.environ["JWT_SECRET_KEY"], algorithm=os.environ["JWT_ALGORITHM"])


async def get_organization_token(
    api_version: ApiVersion,
    vendor: str = Path(description="Vendor name", example="livechat"),
    organization: str = Path(
        description="Organization within vendor", example="f1ac8408-27b2-465e-89c6-b8708bfc262c"
    ),
    password: str = Body(..., description="This is for staff use"),
):
    if password != os.environ["AUTH_COLLECTION_PASSWORD"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token({"organization": organization, "vendor": vendor})
    return {"access_token": access_token}


async def get_livechat_token(api_version: ApiVersion, livechat_token: str = Body(...)):
    headers = {
        "Authorization": f"Bearer {livechat_token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    response = requests.get(f"https://accounts.livechatinc.com/v2/info", headers=headers)
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    resp_data = response.json()
    access_token = create_access_token(
        data={"organization": resp_data["organization"], "vendor": "livechat"}
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def validate_organization_scope(
    vendor: str = Path(description="Vendor name", example="livechat"),
    organization: str = Path(
        description="Organization within vendor", example="f1ac8408-27b2-465e-89c6-b8708bfc262c"
    ),
    token: str = Depends(oauth2_scheme),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            os.environ["JWT_SECRET_KEY"],
            algorithms=[os.environ["JWT_ALGORITHM"]],
        )
        organization_token: str = payload.get("organization")
        vendor_token: str = payload.get("vendor")
        if (
            organization_token is None
            or organization_token != organization
            or vendor_token != vendor
        ):
            raise credentials_exception
    except JWTError:
        raise credentials_exception
