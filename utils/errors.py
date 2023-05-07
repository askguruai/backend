class BasicError(Exception):
    def __init__(self):
        self.message = "An error occured"

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class InvalidDocumentIdError(BasicError):
    def __init__(self, message: str = ""):
        self.message = message


class RequestDataModelMismatchError(BasicError):
    def __init__(self, message: str = ""):
        self.message = message


class CoreMLError(BasicError):
    def __init__(self, message: str = ""):
        self.message = message


class CollectionDoesNotExist(BasicError):
    def __init__(self, message: str = "Requested collection does not exist!"):
        self.message = message


class SecurityGroupError(BasicError):
    def __init__(self, message: str = "Security group error"):
        self.message = message


class DatabaseError(BasicError):
    def __init__(self, message: str = ""):
        self.message = message
