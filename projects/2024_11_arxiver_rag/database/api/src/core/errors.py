class BaseException(Exception):
    CODE = 1000

    def __init__(self, message):
        self.message = message


class NotFoundException(BaseException):
    def __init__(self, message):
        self.message = message