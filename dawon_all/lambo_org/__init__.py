"""lambo_org — a cleaner retrieval baseline.

Sibling of the original ``lambo`` package. This variant removes our custom
"anchor summary generation" contribution and replaces it with a plain
chunk-embedding + cosine-similarity retrieval baseline. Downstream
composition/generation remains unchanged, and the original ``lambo`` package
is not modified.
"""

from .config import LamboOrgConfig  # noqa: F401
