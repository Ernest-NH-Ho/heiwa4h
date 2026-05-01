import torch.nn as nn
import torch

def zero_initialize(self, module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding)):
    with torch.no_grad():
        for m in self.modules():
            if isinstance(m, module_types):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.fill_(0)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.fill_(0)

def orthogonal_initialize(self, module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding)):
    with torch.no_grad():
        for m in self.modules():
            if isinstance(m, module_types):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.orthogonal_(m.weight)
                    print(f"{m}.weight has been orthogonal-initialized.")
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.fill_(0)

def scale_parms(self, module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding),):
     with torch.no_grad():
         for m in self.modules():
            if isinstance(m, module_types):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.mul_(factor)
                    
def xavier_initialize(self, module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding)):
    with torch.no_grad():
        for m in self.modules():
            if isinstance(m, module_types):
                if hasattr(m, 'weight') and m.weight is not None:
                    gain = getattr(m, "gain", 1.0) # fall back to the default gain (1.0) if the attr isn't set
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                    if gain != 1.0:
                        print(f"{m}.gain change to {gain}")
                if hasattr(m, 'bias') and m.bias is not None:
                    bias = getattr(m, "Bias", 0)
                    m.bias.fill_(bias)
                    if bias != 0:
                        print(f"{m}.bias change to {bias}")
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None: 
                    m.bias.fill_(0)

def switch_grad(
    model: nn.Module,
    requires_grad: bool,
    module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding, nn.LayerNorm),
) -> None:
    """ Toggle requires_grad for parameters
      • that belong to modules of the specified types  (set module_types=None for *all* modules)
      • and whose fully-qualified names match `name_list`
        – pass "all" to affect every parameter under the selected modules
        – otherwise provide an iterable of names
    Everything else is set to the opposite flag so that the final state is
    unambiguous (mirrors the behaviour of your original `turn_on_grad_by_name`).    """

    for mname, module in model.named_modules():
        if (module_types is None) or isinstance(module, module_types):
            for pname, param in module.named_parameters(recurse=False):
                param.requires_grad =  requires_grad # if hit else not requires_grad
