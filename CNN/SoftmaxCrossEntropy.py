import math

class SoftmaxCrossEntropy:
    def __init__(self):
        # Stores softmax output and the index of the correct class
        self.probs = None
        self.target_index = None

    def forward(self, logits, target_index):
        self.target_index = target_index
        max_logit = 0.0
        if logits:
            max_logit = max(logits)

        exps = [math.exp(l - max_logit) for l in logits]
        total_sum_exps = sum(exps)
        if total_sum_exps == 0:
            self.probs = [1.0 / len(logits) if logits else 0.0 for _ in logits]
        else:
            self.probs = [e / total_sum_exps for e in exps]

        if not self.probs or target_index >= len(self.probs):
            return float('inf')

        loss_val = -math.log(self.probs[target_index] + 1e-10)
        return loss_val

    # Returns the gradient of the loss with respect to each input logit
    def backward(self):
        grad_dL_dlogits = list(self.probs)
        if self.target_index < len(grad_dL_dlogits):
            grad_dL_dlogits[self.target_index] -= 1
        return grad_dL_dlogits
