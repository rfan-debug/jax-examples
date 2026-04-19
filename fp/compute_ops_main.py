

from fp.compute_ops import Functor, Maybe, Batched, PyTree



def demo_maybe():
    """Maybe Applicative: 可失败的并行计算"""
    print("=" * 60)
    print("Maybe Applicative — 可失败计算的组合")
    print("=" * 60)

    # 两个独立的查询，任一失败则整体失败
    x = Maybe.just(3)
    y = Maybe.just(5)
    n = Maybe.nothing()

    add = lambda a, b: a + b

    # liftA2: 把普通二元函数提升到 Maybe 里
    print(f"liftA2(add, Just(3), Just(5))  = {x.liftA2(add, y)}")
    print(f"liftA2(add, Just(3), Nothing)  = {x.liftA2(add, n)}")
    print(f"liftA2(add, Nothing, Just(5))  = {n.liftA2(add, y)}")
    print()

    # 对比 Monad 的区别：
    # Applicative: 两个查询可以并行，因为 y 不依赖 x 的结果
    # Monad:       y = f(x) 时必须等 x 完成才能开始 y
    print("关键洞察：liftA2(f, x, y) 中 x 和 y 互不依赖")
    print("=> 可以并行求值 x 和 y，然后合并")
    print("=> Monad 的 x >>= (\\a -> f a y) 则必须先求 x")
    print()


def demo_batched():
    """Batched Applicative: vmap 的本质"""
    print("=" * 60)
    print("Batched Applicative — vmap 的纯 Python 实现")
    print("=" * 60)

    xs = Batched([1.0, 2.0, 3.0, 4.0])
    ys = Batched([10.0, 20.0, 30.0, 40.0])

    # fmap = vmap(f) 单参数
    print(f"fmap (x -> x^2) {xs}")
    print(f"  = {xs.fmap(lambda x: x ** 2)}")
    print()

    # liftA2 = vmap(f) 多参数 (in_axes=(0,0))
    print(f"liftA2 (+) {xs} {ys}")
    print(f"  = {xs.liftA2(lambda x, y: x + y, ys)}")
    print()

    # 链式组合 — 函数式管线
    result = (
        xs
        .fmap(lambda x: x * 2)             # vmap(lambda x: x*2)
        .liftA2(lambda a, b: a + b, ys)     # vmap(lambda a,b: a+b)
        .fmap(lambda x: x ** 0.5)           # vmap(sqrt)
    )
    print(f"pipeline: sqrt(2*x + y) = {result}")
    print()
    print("关键洞察：每一步的 batch 结构不变（长度=4）")
    print("=> Applicative 保证输出形状 = 输入形状")
    print("=> XLA 编译器可以在编译期确定所有 buffer 大小")
    print()


def demo_pytree():
    """PyTree Applicative: JAX 参数更新的本质"""
    print("=" * 60)
    print("PyTree Applicative — JAX tree_map 的本质")
    print("=" * 60)

    import math

    # 模拟神经网络参数
    params = PyTree({'w1': 1.0, 'b1': 0.5, 'w2': -0.3, 'b2': 0.1})
    grads  = PyTree({'w1': 0.1, 'b1': 0.02, 'w2': -0.05, 'b2': 0.01})

    lr = 0.01

    # SGD update = tree_map(lambda p, g: p - lr * g, params, grads)
    # 这就是 liftA2！
    new_params = params.liftA2(lambda p, g: p - lr * g, grads)
    print(f"params:     {params}")
    print(f"grads:      {grads}")
    print(f"SGD update: {new_params}")
    print()

    # Adam 也是 Applicative — 多个 PyTree 的 zipWith
    # liftA3(adam_step, params, m, v)
    # 每个叶子的更新互不依赖 => 可以并行
    print("关键洞察：SGD/Adam 的参数更新是 Applicative 的")
    print("=> 每个参数的更新互不依赖（没有 monadic bind）")
    print("=> 可以在多个设备上分片并行更新")
    print()


def demo_applicative_vs_monad():
    """核心区别的直观演示"""
    print("=" * 60)
    print("Applicative vs Monad — 并行性的根本区别")
    print("=" * 60)

    # ---- Applicative: 静态结构，可并行 ----
    print("\n[Applicative] 独立计算的合并:")
    print("  liftA2(f, compute_x(), compute_y())")
    print("  compute_x 和 compute_y 可以并行执行")
    print("  因为 y 不需要等 x 的结果")
    print()

    # 用 Batched 演示
    xs = Batched([1, 2, 3])
    ys = Batched([4, 5, 6])

    # 这三个加法完全独立，可以同时计算
    result = xs.liftA2(lambda x, y: x + y, ys)
    print(f"  Batched [1,2,3] + [4,5,6] = {result}")
    print(f"  三个加法互不依赖 => 三路并行")

    # ---- Monad: 动态结构，必须顺序 ----
    print("\n[Monad] 顺序依赖的链式计算:")
    print("  x >>= (\\a -> if a > 0 then Just(a+1) else Nothing)")
    print("  第二步的结构（Just vs Nothing）依赖第一步的值")
    print("  => 无法并行，必须等第一步完成")
    print()

    # 用 Maybe 演示
    def safe_div(x, y):
        return Maybe.just(x / y) if y != 0 else Maybe.nothing()

    print(f"  safe_div(10, 2) = {safe_div(10, 2)}")
    print(f"  safe_div(10, 0) = {safe_div(10, 0)}")
    print()

    # Monadic chain: 每一步的存在性依赖前一步
    # 这不能用 Applicative 表达，因为后续计算的"形状"是动态的
    print("  Monadic chain: safe_div(10, x) >>= safe_div(_, y)")
    print("  如果第一步返回 Nothing，第二步根本不存在")
    print("  => 结构依赖值 => Monad，非 Applicative")
    print()

    # ---- 对应到 JAX ----
    print("[JAX 对应]")
    print("  Applicative: vmap, tree_map, pmap")
    print("    => 计算图形状编译期确定 => XLA 可优化")
    print("  Monad: lax.scan, lax.while_loop, lax.cond")
    print("    => 计算图形状运行时确定 => 需要动态调度")
    print()


def demo_jax_applicative():
    """用真正的 JAX 展示 Applicative 结构"""
    print("=" * 60)
    print("JAX 中的 Applicative — 实际计算")
    print("=" * 60)

    try:
        import jax
        import jax.numpy as jnp
        import jax.tree_util as tree
    except ImportError:
        print("JAX not installed, skipping JAX demo")
        return

    # 1. vmap 就是 Batched Applicative 的 fmap
    print("\n[1] vmap = Batched.fmap")
    f = lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2  # 应该恒等于 1
    xs = jnp.linspace(0, 3.14, 5)
    print(f"  vmap(sin²+cos²)({xs}) = {jax.vmap(f)(xs)}")

    # 2. 多参数 vmap 就是 liftA2
    print("\n[2] vmap 多参数 = Batched.liftA2")
    g = lambda x, y: x ** 2 + y ** 2
    xs = jnp.array([1.0, 2.0, 3.0])
    ys = jnp.array([4.0, 5.0, 6.0])
    print(f"  vmap(x²+y²)({xs}, {ys}) = {jax.vmap(g)(xs, ys)}")

    # 3. tree_map 就是 PyTree Applicative
    print("\n[3] tree_map = PyTree.fmap / PyTree.liftA2")
    params = {'layer1': {'w': jnp.ones((2, 3)), 'b': jnp.zeros(2)},
              'layer2': {'w': jnp.ones((1, 2)), 'b': jnp.zeros(1)}}
    grads  = tree.tree_map(lambda p: p * 0.1, params)  # 模拟梯度

    lr = 0.01
    new_params = tree.tree_map(lambda p, g: p - lr * g, params, grads)
    print(f"  SGD update via tree_map:")
    print(f"    layer1.w shape: {new_params['layer1']['w'].shape}")
    print(f"    layer1.b: {new_params['layer1']['b']}")

    # 4. Applicative vs Monad 在 JAX 中的性能差异
    print("\n[4] Applicative (vmap) vs Monad (scan) 性能差异")

    # Applicative: 独立计算，vmap 可以一次性并行
    @jax.jit
    def applicative_norm(xs):
        return jax.vmap(jnp.linalg.norm)(xs)

    # Monad: 累积计算，scan 必须顺序执行
    @jax.jit
    def monadic_cumsum(xs):
        def step(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        _, result = jax.lax.scan(step, jnp.zeros_like(xs[0]), xs)
        return result

    batch = jax.random.normal(jax.random.PRNGKey(0), (100, 64))

    # 触发编译
    _ = applicative_norm(batch)
    _ = monadic_cumsum(batch)

    # 实际计算
    norms = applicative_norm(batch)
    cumsums = monadic_cumsum(batch)
    print(f"  vmap(norm) 100 vectors: shape={norms.shape}")
    print(f"  scan(cumsum) 100 vectors: shape={cumsums.shape}")
    print(f"  vmap 的 100 个计算互不依赖 => GPU 并行")
    print(f"  scan 的第 i 步依赖第 i-1 步 => 顺序执行")

    # 5. Applicative 组合的优势：变换可以随意嵌套
    print("\n[5] Applicative 变换的自由组合")
    h = lambda x: jnp.sum(x ** 2)

    # vmap . grad = Batched . Grad 两个 functor 的组合
    # functor 的组合还是 functor（范畴论保证）
    batch_grad = jax.vmap(jax.grad(h))
    xs = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"  vmap(grad(sum(x²))):")
    print(f"    input shape:  {xs.shape}")
    print(f"    output:       {batch_grad(xs)}")
    print(f"    = 2*x (解析梯度), 对 batch 中每个向量独立计算")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_maybe()
    demo_batched()
    demo_pytree()
    demo_applicative_vs_monad()
    demo_jax_applicative()