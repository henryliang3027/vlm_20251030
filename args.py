# *args example
def example_function(*args):
    for arg in args:
        print(arg)

example_function("Hello", "World", 42)



# **kwargs example
# 
def example_function_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

example_function_kwargs(name="Alice", age=30, city="Wonderland")