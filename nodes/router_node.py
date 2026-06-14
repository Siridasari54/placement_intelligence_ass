def router_node(state):
    query = state["query"].lower()

    if any(word in query for word in [
        "cgpa",
        "student",
        "package",
        "backlog"
    ]):
        return {"route": "database"}

    if any(word in query for word in [
        "latest",
        "today",
        "current",
        "ipl",
        "stock"
    ]):
        return {"route": "web"}

    return {"route": "rag"}