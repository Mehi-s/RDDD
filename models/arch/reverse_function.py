
import torch
from typing import Any, Iterable, List, Tuple, Callable
import torch.distributed as dist

def get_gpu_states(fwd_gpu_devices) -> Tuple[List[int], List[torch.Tensor]]:
  """Gets the random number generator states for a list of GPU devices.

  Args:
    fwd_gpu_devices: A list of GPU device IDs.

  Returns:
    A tuple containing a list of GPU device IDs and a list of their RNG states.
  """
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states

def get_gpu_device(*args):
  """Gets the unique GPU devices used by the input tensors.

  Args:
    *args: A variable number of input tensors.

  Returns:
    A list of unique GPU device IDs.
  """

    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices

def set_device_states(fwd_cpu_state, devices, states) -> None:
  """Sets the RNG states for CPU and GPU devices.

  Args:
    fwd_cpu_state: The CPU RNG state.
    devices: A list of GPU device IDs.
    states: A list of GPU RNG states.
  """
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
  """Detaches input tensors and sets requires_grad to True.

  Args:
    inputs: A tuple of tensors or other types.

  Returns:
    A tuple of detached tensors with requires_grad set to True,
    or the original input if not a tensor.
  """
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def get_cpu_and_gpu_states(gpu_devices):
  """Gets the CPU and GPU RNG states.

  Args:
    gpu_devices: A list of GPU device IDs.

  Returns:
    A tuple containing the CPU RNG state and a list of GPU RNG states.
  """
    return torch.get_rng_state(), get_gpu_states(gpu_devices)

class ReverseFunction(torch.autograd.Function):
  """Custom autograd function for reversible computations.

  This function allows for memory-efficient backpropagation by recomputing
  intermediate activations during the backward pass instead of storing them.
  """
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
      """Forward pass for the reversible function.

      Args:
        ctx: Context object to save tensors and information for backward pass.
        run_functions: A list or tuple of functions to be executed sequentially.
        alpha: A list or tuple of alpha values (scaling factors for shortcuts).
        *args: Input arguments for the functions.

      Returns:
        The output of the sequence of functions.
      """
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions  = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0  = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0*alpha0
            ctx.cpu_states_1, ctx.gpu_states_1  = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1*alpha1
            ctx.cpu_states_2, ctx.gpu_states_2  = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2*alpha2
            ctx.cpu_states_3, ctx.gpu_states_3  = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3*alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1 ,c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
      """Backward pass for the reversible function.

      This method recomputes the activations from the previous layers and
      calculates the gradients.

      Args:
        ctx: Context object with saved tensors and information from forward pass.
        *grad_outputs: Gradients of the outputs from the next layer.

      Returns:
        Gradients for the inputs of the reversible function.
      """
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            
            g3_up = g3_right
            g3_left = g3_up*alpha3 ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)                    
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1/alpha3)*(c3 - oup3) ## feature reverse
            g2_up = g2_right+ c2.grad
            g2_left = g2_up*alpha2 ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)          
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left*alpha3 ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1/alpha2)*(c2 - oup2) ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right+c1.grad
            g1_left = g1_up*alpha1 ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)     
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left*alpha2 ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1/alpha1)*(c1 - oup1) ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up*alpha0 ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left ## Fusion
            
            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)     
            oup0 = l0(x, c1_left)            
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left*alpha1 ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1/alpha0)*(c0 - oup0) ## feature reverse
            gx_up = x.grad ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left*alpha0 ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
        
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


