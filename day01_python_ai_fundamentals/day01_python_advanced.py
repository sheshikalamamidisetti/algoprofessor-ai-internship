"""
Day 01 — Python Advanced Syntax
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
"""

# List Comprehension
squares = [x**2 for x in range(10)]
print("Squares:", squares)

# Lambda Function
multiply = lambda x, y: x * y
print("Multiply:", multiply(3, 4))

# Generator
def number_generator(n):
    for i in range(n):
        yield i * 2

gen = number_generator(5)
print("Generator:", list(gen))

# Decorator
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Sheshikala")

# Dictionary Comprehension
data = {"a": 1, "b": 2, "c": 3}
squared = {k: v**2 for k, v in data.items()}
print("Dict comprehension:", squared)

print("\nDay 01 Python advanced syntax completed.")

