import contextlib

from fastapi import FASTAPI
from pydantic import BaseModel
from typing import List

class SpamCollector(BaseModel):
    