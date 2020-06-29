#Uses python3

import sys
import queue
import threading
# from typing import List, Union

sys.setrecursionlimit(10 ** 6)  # max depth of recursion
threading.stack_size(2 ** 27)  # new thread will get stack of such size


from collections import namedtuple

Node = namedtuple('Node', ['nodeID', 'dist'])

class MinHeap():
    def __init__(self, s: int, m: int, dist):#: List[int]):
        self.s = s # origin node
        self.dist = dist
        self.n = len(dist)
        self.maxSize = m + self.n
        self.size = 0
        self.heap = [None for __ in range(self.maxSize)] # heap stores node #s, with heap key as dist[node]
        # heap may have duplicate nodes, so track if nodes is processed already
        # less tricky than trying to find duplicate nodes in heap
        self.extractedFromHeap = [False for __ in range(self.n)]
        self.init_heap()

    def init_heap(self):
        self.dist[self.s] = 0
        for node in range(self.n):
            nodeToInsert = Node(nodeID=node, dist=self.dist[node])
            self.insert(nodeToInsert)

    """
    def print_heap(self):
        print(f"heap size={self.size}")
        for index in range(self.size):
            print(f"index={index}; node={self.heap[index]}; dist[node] = {self.dist[self.heap[index].nodeID]}")

        print()
    """

    # input heap index, output parent heap index
    def parent(self, i: int): #-> Union[int, None]:
        if i > 0:
            return (i - 1) // 2
        else:
            return None

    # input heap index, output left child heap index
    def left_child(self, i: int) -> int:
        return 2 * i + 1

    # input heap index, output right child heap index
    def right_child(self, i: int) -> int:
        return 2 * i + 2

    # input heap index, sift up a node at heap index based on distance of the node
    def siftUp(self, i: int):

        # print(f"siftUp i={i} self.parent(i)={self.parent(i)}")
        # print(f"start siftUp of node {self.heap[i]}")
        # self.print_heap()

        while i > 0 and self.heap[self.parent(i)].dist > self.heap[i].dist: # min heap, so disallowed for parent > child
            self.heap[self.parent(i)], self.heap[i] = self.heap[i], self.heap[self.parent(i)]
            i = self.parent(i)
            # print("after swap in siftUp")
            # self.print_heap()
        """
        while i > 0 and self.dist[self.heap[self.parent(i)]] > self.dist[self.heap[i]]: # min heap, so disallowed for parent > child
            self.heap[self.parent(i)], self.heap[i] = self.heap[i], self.heap[self.parent(i)]
            i = self.parent(i)
            print("after swap in siftUp")
            self.print_heap()
        """
    """        
    def siftDown(self, i: int):
        minIndex = i
        leftIndex = self.left_child(i)
        if leftIndex < self.size and self.dist[self.heap[leftIndex]] < self.dist[self.heap[minIndex]]: # min heap, so sift larger parent to left child
            minIndex = leftIndex
        rightIndex = self.right_child(i)
        if rightIndex < self.size and self.dist[self.heap[rightIndex]] < self.dist[self.heap[minIndex]]: # min heap, so sift larger parent to right child
            minIndex = rightIndex
        if i != minIndex:
            self.heap[i], self.heap[minIndex] = self.heap[minIndex], self.heap[i]
            self.siftDown(minIndex)
    """

    def siftDown(self, i: int):
        minIndex = i
        leftIndex = self.left_child(i)
        if leftIndex < self.size and self.heap[leftIndex].dist < self.heap[minIndex].dist: # min heap, so sift larger parent to left child
            minIndex = leftIndex
        rightIndex = self.right_child(i)
        if rightIndex < self.size and self.heap[rightIndex].dist < self.heap[minIndex].dist: # min heap, so sift larger parent to right child
            minIndex = rightIndex
        if i != minIndex:
            self.heap[i], self.heap[minIndex] = self.heap[minIndex], self.heap[i]
            self.siftDown(minIndex)

    def insert(self, node: Node):
        if self.size == self.maxSize - 1:
            assert False
        # print(f"before insert of node {node}")
        # self.print_heap()
        self.size += 1
        self.heap[self.size - 1] = node
        # print(f"after insert of node {node}")
        # self.print_heap()
        self.siftUp(self.size - 1)

    def extractMin(self) -> int:
        if self.size == 0:
            assert False
        result = self.heap[0].nodeID
        if self.size > 1:
            self.heap[0] = self.heap[self.size - 1]
        else:
            self.heap[0] = None
        self.size -= 1
        if self.size > 1:
            self.siftDown(0)
        return result

    def remove(self, i):
        self.dist[self.heap[i]] = float('-inf')
        self.siftUp(i)

    # linear time is too expensive to find node in min heap.
    # instead, add another node to min heap, and sift up.
    # as extractMin removes nodes from min heap, process removed node only if extractedFromHeap[node] == False
    # when removed node is processed, set extractedFromHeap[node] = True
    def changePriority(self, nodeID: int, newDistance: int):
        self.dist[nodeID] = newDistance
        node = Node(nodeID, newDistance)
        self.insert(node)
        # print(f"after insert of node {node}")
        # self.print_heap()

# def distance(adj: List[List[int]], cost: List[List[int]], s: int, t: int, m: int) -> int:
def distance(adj, cost, s: int, t: int, m: int) -> int:

    # if len(adj) == 1:
    #    return 0

    """
    if m == 0:
        if s != t:
            return -1
        elif s == t:
            return 0
    """
    # since, with no edges, and s == t disallowed, there is no path
    if m == 0:
        return -1

    # ruled out by problem
    # if s == t:
        # return 0

    #write your code here
    dist = [float("inf") for __ in range(len(adj))]
    prev = [None for __ in range(len(adj))]

    # print(f"adj=={adj} cost=={cost} dist={dist} prev=={prev}")

    heap = MinHeap(s, m, dist)

    # heap.print_heap()

    while heap.size > 0:

        # heap.print_heap()

        u = heap.extractMin()
        # print("after extractMin")
        # heap.print_heap()
        if heap.extractedFromHeap[u]:
            continue
        else:
            heap.extractedFromHeap[u] = True
        for index, v in enumerate(adj[u]):
            if dist[v] > dist[u] + cost[u][index]:
                dist[v] = dist[u] + cost[u][index]
                prev[v] = u
                # print("before changePriority")
                # heap.print_heap()
                heap.changePriority(v, dist[v])
                # print("after changePriority")
                # heap.print_heap()

    if dist[t] == float("inf"):
        return -1
    else:
        return dist[t]

def main():
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n, m = data[0:2]
    data = data[2:]
    edges = list(zip(zip(data[0:(3 * m):3], data[1:(3 * m):3]), data[2:(3 * m):3]))
    data = data[3 * m:]
    adj = [[] for _ in range(n)]
    cost = [[] for _ in range(n)]
    for ((a, b), w) in edges:
        adj[a - 1].append(b - 1)
        cost[a - 1].append(w)
    s, t = data[0] - 1, data[1] - 1
    print(distance(adj, cost, s, t, m))

if __name__ == '__main__':

    threading.Thread(target=main).start()