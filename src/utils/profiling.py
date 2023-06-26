import torch
import torch.utils.benchmark as benchmark


def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated() / (
        (2**20) * 1000
    ), torch.cuda.memory_cached() / ((2**20) * 1000)


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append(
            {
                "layer_idx": idx,
                "call_idx": call_idx,
                "layer_type": type(self).__name__,
                "exp": exp,
                "hook_type": hook_type,
                "mem_all": mem_all,
                "mem_cached": mem_cached,
            }
        )

    return hook


def _add_memory_hooks(idx, model, mem_log, exp, hr):
    h = model.register_forward_pre_hook(
        _generate_mem_hook(hr, mem_log, idx, "pre", exp)
    )
    hr.append(h)

    h = model.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, "fwd", exp))
    hr.append(h)

    h = model.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, "bwd", exp))
    hr.append(h)


def log_memory(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f"exp_{len(mem_log)}"
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    out = model(inp)
    if type(out) == tuple:
        out = out[0].logits
    loss = out.sum()
    loss.backward()
    [h.remove() for h in hr]
    return mem_log


def benchmark_forward(
    fn, *inputs, min_run_time=0.2, repeats=10, desc="", verbose=True, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")
    t = benchmark.Timer(
        stmt="fn(*inputs, **kwinputs)",
        globals={"fn": fn, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem


def benchmark_memory_bwd(fn, *inputs, desc="", verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for input in inputs:
        input = input.requires_grad_(True)
    torch.cuda.synchronize()
    y = fn(*inputs, **kwinputs)
    y.sum().backward()
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem


def benchmark_backward(
    fn, *inputs, grad=None, repeats=10, desc="", verbose=True, **kwinputs
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    y = fn(*inputs, **kwinputs)
    if not hasattr(y, "shape"):
        y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")
    t = benchmark.Timer(
        stmt="y.backward(grad, retain_graph=True)",
        globals={"y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m
