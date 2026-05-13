"""Generator wrappers — 외부 motion generator의 wrapper.

각 wrapper는 generators.base.Generator 인터페이스를 구현해 canonical
[T, 22, 3] motion + metadata를 반환한다.
"""
from generators.base import Generator

__all__ = ["Generator"]
