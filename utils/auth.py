import os

import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from utils.schemas import (
    AuthenticatedRequest,
    LivechatLoginRequest,
    UploadChatsRequest,
    VendorCollectionRequest,
    VendorCollectionTokenRequest,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_org_collection_token(request: VendorCollectionTokenRequest):
    if request.password != os.environ["AUTH_COLLECTION_PASSWORD"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        {"organization_id": request.organization_id, "vendor": request.vendor}
    )
    return {"access_token": access_token}


def create_access_token(data: dict):
    return jwt.encode(data, os.environ["JWT_SECRET_KEY"], algorithm=os.environ["JWT_ALGORITHM"])


async def login_livechat(request: LivechatLoginRequest):
    headers = {
        "Authorization": f"Bearer {request.livechat_token}",
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
        data={"organization_id": resp_data["organization_id"], "vendor": "livechat"}
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def validate_auth_org_scope(
    user_request: VendorCollectionRequest, token: str = Depends(oauth2_scheme)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, os.environ["JWT_SECRET_KEY"], algorithms=[os.environ["JWT_ALGORITHM"]]
        )
        org_id: str = payload.get("organization_id")
        vendor: str = payload.get("vendor")
        if (
            org_id is None
            or org_id != user_request.organization_id
            or vendor != user_request.vendor
        ):
            raise credentials_exception
    except JWTError:
        raise credentials_exception
