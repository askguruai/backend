class InvalidDocumentIdError(BaseException):
    @staticmethod
    def response(**kwargs):
        response_dict = {"error_msg": "Invalid document ID", "status": "error"}
        for k, v in kwargs.items():
            response_dict.update({k: str(v)})
        return response_dict
