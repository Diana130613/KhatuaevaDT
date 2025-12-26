import unittest
from kmp_search import kmp_search
from string_matching import naive_search, rabin_karp
from z_function import z_search


class TestStringAlgorithms(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            ("abcabc", "abc", [0, 3]),
            ("aaaa", "aa", [0, 1, 2]),
            ("abcdef", "xyz", []),
            ("", "a", []),
            ("abc", "", [0, 1, 2, 3]),
            ("a"*100 + "b", "a"*10 + "b", [90]),  # Худший случай для наивного
        ]

    def test_all_algorithms(self):
        algorithms = [
            ("naive", naive_search),
            ("kmp", kmp_search),
            ("z_search", z_search),
            ("rabin_karp", rabin_karp),
        ]

        for text, pattern, expected in self.test_cases:
            for name, func in algorithms:
                with self.subTest(text=text, pattern=pattern, algorithm=name):
                    result = func(text, pattern)
                    self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
