class BasicError(BaseException):
    def __init__(self):
        self.message = "An error occured"

    def response(self, **kwargs):
        response_dict = {
            "error_msg": self.message,
            "error": self.__class__.__name__,
            "status": "error",
        }
        for k, v in kwargs.items():
            response_dict.update({k: str(v)})
        return response_dict


class InvalidDocumentIdError(BasicError):
    def __init__(self, message: str = None):
        self.message = "Invalid document ID" if message is None else message


class RequestDataModelMismatchError(BasicError):
    def __init__(self, message: str = None):
        self.message = "Request data did not match data model" if message is not None else message
