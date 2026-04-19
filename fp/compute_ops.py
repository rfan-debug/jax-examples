
from __future__ import annotations
from typing import TypeVar, Generic, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


# ============================================================
# Part 1: 最朴素的 Applicative — 从 Functor 开始
# ============================================================

class Functor(ABC, Generic[A]):
    """
    Functor law:
      fmap id = id
      fmap (f . g) = fmap f . fmap g
    """
    @abstractmethod
    def fmap(self, f: Callable[[A], B]) -> Functor[B]:
        ...


class Applicative(Functor[A]):
    """
    Applicative 在 Functor 之上增加两个操作：
      pure  : a -> F a                    （把值放进容器）
      ap    : F (a -> b) -> F a -> F b    （容器里的函数应用到容器里的值）

    Applicative laws:
      identity:     pure id <*> v = v
      composition:  pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
      homomorphism: pure f <*> pure x = pure (f x)
      interchange:  u <*> pure y = pure ($ y) <*> u
    """
    @staticmethod
    @abstractmethod
    def pure(value: B) -> Applicative[B]:
        ...

    @abstractmethod
    def ap(self, fa: Applicative[A]) -> Applicative[B]:
        """self 包含函数 a->b, fa 包含值 a, 返回包含 b"""
        ...


    def liftA2(self, f: Callable[[A, Any], B], fb: Applicative) -> Applicative[B]:
        """liftA2 f fa fb = fmap f fa <*> fb"""
        curried = lambda a: lambda b: f(a, b)
        return self.fmap(curried).ap(fb)


# Maybe (Optional)

@dataclass
class Maybe(Applicative[A]):
    """
    Maybe a = Nothing | Just a
    经典的可失败计算：任何一步失败，整个链路失败
    """
    _value: A | None
    _is_just: bool

    @staticmethod
    def just(x: A) -> Maybe[A]:
        return Maybe(x, True)

    @staticmethod
    def nothing() -> Maybe[Any]:
        return Maybe(None, False)

    @staticmethod
    def pure(value: B) -> Maybe[B]:
        return Maybe.just(value)

    def fmap(self, f: Callable[[A], B]) -> Maybe[B]:
        if self._is_just:
            return Maybe.just(f(self._value))
        return Maybe.nothing()

    def ap(self, fa: Maybe[A]) -> Maybe[B]:
        # self 包含函数, fa 包含值
        if self._is_just and fa._is_just:
            return Maybe.just(self._value(fa._value))
        return Maybe.nothing()

    def __repr__(self):
        return f"Just({self._value})" if self._is_just else "Nothing"



@dataclass
class Batched(Applicative[A]):
    """
    Batched a = [a]

    这就是 JAX vmap 的本质抽象：
    - pure 把标量广播成 batch
    - fmap 对每个元素独立 map（= vmap(f)）
    - ap 把一批函数逐元素应用到一批值（= vmap 多参数版本）

    关键：所有元素的计算是独立的，可以并行
    这就是为什么 Applicative 比 Monad 更适合并行计算
    """
    _values: list[A]

    @staticmethod
    def pure(value: B) -> Batched[B]:
        # 广播：标量变成单元素 batch
        return Batched([value])

    def fmap(self, f: Callable[[A], B]) -> Batched[B]:
        # = vmap(f)(self._values)
        return Batched([f(x) for x in self._values])

    def ap(self, fa: Batched[A]) -> Batched[B]:
        # self 包含函数 [f1, f2, ...], fa 包含值 [a1, a2, ...]
        # zipWith apply — 这就是 vmap 对多参数的处理
        assert len(self._values) == len(fa._values), "Batch size mismatch"
        return Batched([f(a) for f, a in zip(self._values, fa._values)])

    def __repr__(self):
        return f"Batched({self._values})"



@dataclass
class PyTree(Applicative[A]):
    """
    PyTree 模拟 JAX 的 tree_util
    叶子节点存值，树结构是 Applicative 的"容器形状"

    tree_map(f, tree)          = fmap
    tree_map(f, tree1, tree2)  = liftA2

    关键约束：两个 PyTree 做 ap 时结构必须对齐
    这就是 Applicative（不是 Monad）的标志：
    输出的结构由输入的结构决定，不能动态改变
    """
    _data: dict[str, A]  # 简化：用 flat dict 代替嵌套树

    @staticmethod
    def pure(value: B) -> PyTree[B]:
        return PyTree({'_': value})

    def fmap(self, f: Callable[[A], B]) -> PyTree[B]:
        return PyTree({k: f(v) for k, v in self._data.items()})

    def ap(self, fa: PyTree[A]) -> PyTree[B]:
        # self 的每个叶子是函数，fa 的对应叶子是值
        assert set(self._data.keys()) == set(fa._data.keys()), \
            "Tree structure mismatch (Applicative requires aligned shapes)"
        return PyTree({k: self._data[k](fa._data[k]) for k in self._data})

    def __repr__(self):
        return f"PyTree({self._data})"