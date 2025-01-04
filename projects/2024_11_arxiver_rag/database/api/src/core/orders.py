from typing import Literal


class BaseOrder:
    order_by: str
    direction: Literal["asc", "desc"]

    def __init__(self, order_by: str, direction: Literal["asc", "desc"]) -> None:
        self.order_by = order_by
        self.value = direction

    @staticmethod
    def is_type(field_name: str) -> bool:
        return False
