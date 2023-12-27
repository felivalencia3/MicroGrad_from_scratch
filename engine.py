class Value:
    def __init__(self, data, _children=(), _op='') -> None:
        self.data = data
        self.prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
val = a * b + c
print(val._op)
