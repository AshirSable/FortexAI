# Holds the pipeline that main.py builds at startup.
# main.py fills these in; screen.py reads them. (Avoids a circular import.)
pipeline = None
semantic_search = None
