from typing import Any, Dict, List, Union


class BaseFilter:
    field_name: str
    value: Union[Any, List[Any], Dict[str, Any]]

    def __init__(
        self, field_name: str, value: Union[Any, List[Any], Dict[str, Any]]
    ) -> None:
        self.field_name = field_name
        self.value = value

    @classmethod
    def is_type(cls, field_name: str) -> bool:
        return False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BaseFilter":
        return BaseFilter(data["field_name"], data["value"])
