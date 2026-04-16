"""dawonv9 — v8 pipeline + Phase-2 cross-document citation injection.

Key addition over v8
--------------------
* ``enrichment/ref_parser.py``: extracts individual paper titles from
  bibliography anchors using quoted-title and unquoted-title patterns.
* ``enrichment/cross_doc_anchor_enricher.py``: after AnchorAgent processes
  all documents, matches extracted reference titles against other docs'
  titles and injects ``cross_doc_cites`` hints into matching anchors.
* ``enrichment/cross_doc_matcher.py``: extended with prefix_match and
  first_word_match scoring so variant-subtitle papers (e.g. two HuBERT
  papers) are matched correctly.
* ``agents/anchor_agent.py``: calls Phase 2 after per-doc enrichment.
* ``agents/doc_refine_agent.py``: surfaces ``★ CITES: DOC1, DOC3`` in the
  doc_map so the LLM is guided to the right anchor without needing to do
  title matching itself.
"""
