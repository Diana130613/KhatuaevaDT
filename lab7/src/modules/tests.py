import unittest
from heap import Heap
from heapsort import heapsort, heapsort_inplace
from priority_queue import PriorityQueue


class TestHeap(unittest.TestCase):
    def test_heap_ops(self):
        """Вставка, извлечение: min-heap."""
        heap = Heap(is_min=True)
        data = [5, 2, 8, 3, 10, 1]
        for x in data:
            heap.insert(x)
        self.assertEqual(heap.extract(), 1)
        self.assertEqual(heap.extract(), 2)
        self.assertTrue(heap.is_heap())

    def test_heap_build(self):
        """ Построение кучи из массива."""
        heap = Heap(is_min=False)
        array = [1, 5, 3, 8, 4]
        heap.build_heap(array)
        self.assertTrue(heap.is_heap())

    def test_heapsort(self):
        """Проверка Heapsort."""
        array = [5, 1, 7, 3, 2]
        res = heapsort(array)
        self.assertEqual(res, sorted(array))

    def test_heapsort_inplace(self):
        """In-place Heapsort."""
        array = [5, 12, 1, 7, 3, 2]
        arr = array[:]
        heapsort_inplace(arr)
        self.assertEqual(arr, sorted(array))

    def test_priority_queue(self):
        """Приоритетная очередь."""
        pq = PriorityQueue()
        pq.enqueue("low", 3)
        pq.enqueue("high", 1)
        pq.enqueue("medium", 2)
        self.assertEqual(pq.dequeue(), "high")
        self.assertEqual(pq.dequeue(), "medium")
        self.assertEqual(pq.dequeue(), "low")


if __name__ == '__main__':
    unittest.main(verbosity=2)
