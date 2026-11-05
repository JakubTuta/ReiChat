from typing import Annotated

from pydantic import BaseModel, StringConstraints

# DRF's CharField trimmed whitespace and refused a blank value; text fields in
# request bodies keep doing both.
TrimmedStr = Annotated[str, StringConstraints(strip_whitespace=True)]
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def reject_explicit_nulls[M: BaseModel](model: M, *, nullable: frozenset[str] = frozenset()) -> M:
    """For partial updates: a field left out means "leave it", but one sent as
    null means "clear it", which only some fields can do. Call from an
    after-validator with the names that may be cleared."""
    for name in model.model_fields_set - nullable:
        if getattr(model, name) is None:
            raise ValueError(f"{name} can't be empty")
    return model
