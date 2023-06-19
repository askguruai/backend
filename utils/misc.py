from utils.errors import SecurityGroupError
from utils import CONFIG


def int_list_encode(group_list: list) -> int:
    if len(group_list) == 0:
        return 2**63 - 1
    n = 0
    for gr in group_list:
        if gr < 0 or gr > 63:
            raise SecurityGroupError(f"Invalid security group code: {gr}. Value must be between 0 and 63")
        n |= 1 << gr
    return n
