# TODO
- [ ] expand source_type
    - [ ] web
    - [ ] git
    - [ ] table
    - [ ] pdf
- [ ] should be async


MATCH (n)-[r]->(m)
RETURN n, r, m LIMIT 300

MATCH (n)
DETACH DELETE n
