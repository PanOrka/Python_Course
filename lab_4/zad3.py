from random import randint


class Node(object):
    def __init__(self, value, prev):
        self.next = []
        self.value = value
        self.prev = prev


    def add(self, new_node):
        self.next += [new_node]


def gen_tree(height, root, n):
    if height != 1:
        anc_n = randint(1, 5) # ustalam max 5 potomkow
        for i in range(anc_n):
            n[0] += 1
            root.add(Node(n[0], root))
            gen_tree(height-1, root.next[i], n)


def dfs(tree):
    visited = []
    actual = tree
    visited.append(actual)
    yield actual.value
    while actual != None:
        for i in actual.next:
            if not i in visited:
                actual = i
                visited.append(actual)
                yield actual.value
                break
        else:
            actual = actual.prev


def bfs(tree):
    queue = []
    queue.append(tree)
    while len(queue) != 0:
        actual = queue.pop(0)
        for i in actual.next:
            queue.append(i)
        yield actual.value


if __name__ == "__main__":
    tree = Node(1, None)
    gen_tree(5, tree, [1])
    print("DFS:")
    for x in dfs(tree):
        print(x)
    print("============")

    print("BFS:")
    for x in bfs(tree):
        print(x)
    print("============")