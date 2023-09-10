from utils.errors import SecurityGroupError


def int_list_encode(group_list: list | None) -> int:
    if group_list is None or len(group_list) == 0:
        return 2**63 - 1
    n = 0
    for gr in group_list:
        if gr < 0 or gr > 63:
            raise SecurityGroupError(f"Invalid security group code: {gr}. Value must be between 0 and 63")
        n |= 1 << gr
    return n


def decode_security_code(n):
    if n == 2**63 - 1:
        return None  # idk if that's misleading though
    return [i for i, b in enumerate(bin(n)[:1:-1]) if b == "1"]


class AsyncIterator:
    def __init__(self, list):
        self.list = list
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.list):
            result = self.list[self.index]
            self.index += 1
            return result
        else:
            raise StopAsyncIteration
