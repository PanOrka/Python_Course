from random import randint


def gen_tree(height, root, n):
    root += [n[0]]
    n[0] += 1
    if height == 1:
        root += [None]
        root += [None]
    else:
        nodes_n = randint(1, 3)
        if nodes_n == 1:
            root += [[]]
            root += [None]
            gen_tree(height-1, root[1], n)
        elif nodes_n == 2:
            root += [None]
            root += [[]]
            gen_tree(height-1, root[2], n)
        elif nodes_n == 3:
            root += [[]]
            root += [[]]
            gen_tree(height-1, root[1], n)
            gen_tree(height-1, root[2], n)


def dfs(tree):
    visited = []
    trace = []
    actual = tree
    visited.append(actual)
    trace.append(actual)
    yield actual[0]
    while True:
        if actual[1] != None and (not actual[1] in visited):
            actual = actual[1]
            visited.append(actual)
            trace.append(actual)
            yield actual[0]
        elif actual[2] != None and (not actual[2] in visited):
            actual = actual[2]
            visited.append(actual)
            trace.append(actual)
            yield actual[0]
        else:
            trace.pop()
            if (len(trace) != 0):
                actual = trace[-1]
            else:
                break


def bfs(tree):
    queue = []
    queue.append(tree)
    while len(queue) != 0:
        actual = queue.pop(0)
        for i in actual[1:]:
            if (i != None):
                queue.append(i)
        yield actual[0]


if __name__ == "__main__":
    tree = []
    gen_tree(5, tree, [1])
    print(tree)
    print("DFS:")
    for x in dfs(tree):
        print(x)
    print("============")

    print("BFS:")
    for x in bfs(tree):
        print(x)
    print("============")