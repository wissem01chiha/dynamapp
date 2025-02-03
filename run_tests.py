import unittest
import os

def run_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    exit(run_tests())
