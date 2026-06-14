from safety.conflict_detector import ConflictDetector

detector = ConflictDetector()

def conflict_node(state):
    conflicts = detector.check_conflict(
        state["documents"]
    )

    if conflicts:
        print("Conflict detected")

    return {}