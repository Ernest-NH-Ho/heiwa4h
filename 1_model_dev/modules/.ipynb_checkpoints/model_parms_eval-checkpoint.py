import torch.nn as nn
import torch

def check_parms(self, module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding),):
    with torch.no_grad():
        for m in self.modules():
            if isinstance(m, module_types):
                if hasattr(m, 'weight') and m.weight is not None:
                    print(f"{m}", m.weight.mean().item())
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    print(f"{m}", m.weight.mean().item())

# ------------------------------------------------------------------
def check_grad(
    model: nn.Module,
    requires_grad: bool,
    module_types: tuple[nn.Module] | None = (nn.Linear, nn.Embedding, nn.LayerNorm),
) -> None:
    """ Sanity-check that the requires_grad flags match the expectation.
    Raises RuntimeError at the first mismatch encountered.     """

    for mname, module in model.named_modules():
        if (module_types is None) or isinstance(module, module_types):
            for pname, param in module.named_parameters(recurse=False):
                if param.requires_grad != requires_grad:
                    raise RuntimeError(
                        f"[ERROR] {mname}.{pname}: requires_grad={param.requires_grad} "
                        f"(expected {requires_grad})"
                    )

# ------------------------------------------------------------------
class Orthogonal_Loss:
    epsilon = 1e-12
    
    @staticmethod
    def single_W(w):
        """        Compute the orthogonal regularization loss for a weight tensor.
        For weights with more than 2 dimensions (e.g. convolutional filters),
        we first flatten the weight to shape (out_channels, -1).
        The loss is defined as ||W^T W - I||_F^2.        """
        w_norm = w.norm(dim=1, keepdim=True) + Orthogonal_Loss.epsilon
        temp = w/w_norm
        return torch.abs(temp @ temp.transpose(0, -1) - torch.eye(temp.size(0), device='cuda')).mean(dim=1) * w_norm.squeeze()

    @staticmethod
    def Q_K(Q, K):
        QK = Q * K
        QK_norm = QK.norm(dim=1, keepdim=True) + Orthogonal_Loss.epsilon
        QK = QK/QK_norm
        return torch.abs(torch.sum(QK, dim=1))

    @classmethod
    def calculate(cls, train_valid_test:str, update_grad:bool, mode='Single_W', parm1=None, parm2=None):
        # train_valid_test, name in layer_targets, model_config._data[compo]['ortho'], [param]
        if parm1 != None:
            if parm2 == None:
                if mode == 'Single_W':
                    loss = cls.single_W(parm1)

            else:
                loss = cls.Q_K(parm1, parm2)
    
            if update_grad and train_valid_test=='train':
                cls.update_grad(loss, parm1)
                if parm2 != None:
                    cls.update_grad(loss, parm2)

            # Just for reporting
            return loss.mean().item()
        # Just for reporting
        return None

    @staticmethod
    def update_grad(loss, parm):
        if loss != None and parm != None:
            if parm.grad != None:
                # print(parm.grad)
                while len(loss.size()) < len(parm.size()):
                    loss = loss.unsqueeze(-1)
                parm.grad = parm.grad * (1 + loss / 4)
