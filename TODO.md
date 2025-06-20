# TODO
- [ ] expand source_type
    - [ ] web
    - [ ] git
    - [ ] table
    - [ ] pdf
- [ ] should be async

- [ ] Use olmocr instead of docling + microsoft table extraction.
- [ ] should we still use docling hybrid chunker or is there a better alternative.
- [ ] need an alternative strategy for parsing code files.
    - [ ] tried treesitter
        - [ ] perhaps combine with LSP?
    - [ ] perhaps use LLM
    - [ ] how to capture semantic meaning between multiple files on a large project.


MATCH (n)-[r]->(m)
RETURN n, r, m LIMIT 300

MATCH (n)
DETACH DELETE n
