"""
statistics_tools.py - Basic statistical calculations.

This module provides fundamental statistical functions.
"""

import math


def mean(data):
    """Calculate arithmetic mean."""
    return sum(data) / len(data)


def variance(data):
    """Calculate population variance."""
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std_dev(data):
    """Calculate population standard deviation."""
    return math.sqrt(variance(data))


def std_error(data):
    """Calculate standard error of the mean."""
    return std_dev(data) / math.sqrt(len(data))


# Module-level code runs on import
print("Loading statistics_tools module...")

# Test code that only runs when script is executed directly
if __name__ == "__main__":
    test_data = [1, 2, 3, 4, 5]
    print(f"Test mean: {mean(test_data)}")
    print(f"Test std dev: {std_dev(test_data):.3f}")
