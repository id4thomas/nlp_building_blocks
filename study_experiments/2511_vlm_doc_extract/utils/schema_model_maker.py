import json
from typing import Any, Dict, List
from enum import Enum

from pydantic import BaseModel, create_model

from bbox import Bbox

class Bbox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    
    class Config:
        extra="forbid"

def create_dynamic_enum(name: str, values: List[Any]) -> Enum:
    return Enum(name, {str(v): v for v in values})

class SchemaModelMaker:
    @classmethod
    def get_dtype(cls, dtype_name: str, item_specifications: Dict[str, Any]) -> type:
        if dtype_name == "str":
            dtype = str
        elif dtype_name == "int":
            dtype = int
        elif dtype_name == "float":
            dtype = float
        elif dtype_name == "bool":
            dtype = bool
        elif dtype_name == "bbox":
            dtype = Bbox
        elif dtype_name in item_specifications:
            dtype = cls.make(
                specification=item_specifications[dtype_name],
                item_specifications=item_specifications,
                name=dtype_name
            )
        else:
            raise ValueError("dtype {} not supported".format(dtype_name))
        return dtype


    @classmethod
    def make(
        cls,
        specification: Dict[str, Any],
        item_specifications: Dict[str, Any],
        name="model"
    ) -> BaseModel:
        """
        specification must be defined like
        {
            "{KEY_NAME}": {
                    "dim": 0, ## 0: str, 1: List[{dtype}]
                    "dtype": str, # str, int, float, bool
                    "allowed_values": ["a", "b"], ## for enum val
                    "optional": False
            }
        }
        """
        spec_dict = dict()
        for k, v in specification.items():
            dtype = v.get("dtype", str)
            dim = v.get("dim", 0)
            allowed_values = v.get("allowed_values", None)
            default_val = ...

            if isinstance(dtype, str):
                dtype = cls.get_dtype(
                    dtype,
                    item_specifications=item_specifications
                )

            if allowed_values:
                allowed_values = [dtype(v) for v in allowed_values]
                enum = create_dynamic_enum(f"{k}-enum", allowed_values)
                dtype = enum

            ## Dim
            if dim == 1:
                dtype = List[dtype]

            spec_dict[k] = (dtype, default_val)

        model = create_model(name, __config__={"extra": "forbid"}, **spec_dict)
        return model