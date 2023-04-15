import os

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import requests

from utils.schemas import Collection, LivechatLoginRequest, CollectionRequest, UploadChatsRequest

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict):
    return jwt.encode(data, os.environ["JWT_SECRET_KEY"], algorithm=os.environ["JWT_ALGORITHM"])


async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if (
        not form_data.username in [c.value for c in Collection]
        or form_data.password != os.environ["AUTH_COLLECTION_PASSWORD"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})

    return {"access_token": access_token, "token_type": "bearer"}


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
    access_token = create_access_token(data={"organization_id": resp_data["organization_id"]})
    return {"access_token": access_token, "token_type": "bearer"}


async def validate_auth(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, os.environ["JWT_SECRET_KEY"], algorithms=[os.environ["JWT_ALGORITHM"]]
        )
        username: str = payload.get("sub")
        if username is None or username not in [c.value for c in Collection]:
            raise credentials_exception
    except JWTError:
        raise credentials_exception


async def validate_auth_livechat(user_request: CollectionRequest | UploadChatsRequest,
                                 token: str = Depends(oauth2_scheme)):
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
        if org_id is None or org_id != user_request.organization_id:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
