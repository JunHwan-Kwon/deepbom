"""Explicit source-to-export mapping through the common Model IR engine."""
from ._native.core import engine


def correspondence(*, source, target, bindings, documents):
    """Keep declared/observed/ambiguous many-to-many groups and unmapped subjects."""
    return engine("correspondence", {"schema":"deepbom.correspondence_request.v1",
        "source":source,"target":target,"bindings":bindings,"documents":documents})


def mapped_comparison(document):
    """Compare explicit aligned records without inferring equivalence or MAC deltas."""
    return engine("mapped_comparison", document)
