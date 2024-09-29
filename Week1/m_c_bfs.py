from collections import deque

def is_valid(state):
    missionaries, cannibals, _ = state
    if not (0 <= missionaries <= 3 and 0 <= cannibals <= 3):
        return False
    if (missionaries < cannibals and missionaries > 0) or ((3 - missionaries) < (3 - cannibals) and (3 - missionaries) > 0):
        return False
    return True

def get_successors(state):
    missionaries, cannibals, boat = state
    direction = -1 if boat == 1 else 1 
    successors = []
    moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
    for m, c in moves:
        new_state = (missionaries + direction * m, cannibals + direction * c, 1 - boat)
        if is_valid(new_state):
            successors.append(new_state)
    return successors

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        state, path = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = (3, 3, 1)
goal_state = (0, 0, 0)

solution = bfs(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
