from typing import List
from pydantic import BaseModel


class CodebaseFile(BaseModel):
    path: str
    content: str


class CodebasePath(BaseModel):
    path: str
    is_file: bool
    sub_paths: List['CodebasePath'] = []


# Resolve forward reference for recursive type
CodebasePath.model_rebuild()
