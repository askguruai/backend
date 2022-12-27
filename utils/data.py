from pydantic import BaseModel


class TextRequest(BaseModel):
    text_input: str
    query: str
