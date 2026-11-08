"""Per-edge Markov logic network, solved exactly.

An edge with K binary predicates has 2^K worlds (at most 16, or 32 once R5 is added),
so the query posterior is computed by enumeration:

    s(x) = sum_i a_i * l_i * x_i + sum_r w_r * [x satisfies clause r]

l_i is the evidence log-odds of predicate i, a_i a scale and w_r a rule weight. A clause
is a list of (predicate, positive) literals.
"""
import itertools
import torch
import torch.nn as nn


class ExactMLN(nn.Module):
    def __init__(self, preds, rules, query, init_w=1.0):
        super().__init__()
        self.preds = list(preds)
        self.query = self.preds.index(query)
        K = len(self.preds)
        worlds = torch.tensor(list(itertools.product([0, 1], repeat=K)), dtype=torch.float32)  # [W,K]
        sat = []
        for clause in rules.values():
            lit = [worlds[:, self.preds.index(p)] if pos else 1 - worlds[:, self.preds.index(p)]
                   for p, pos in clause]
            sat.append(torch.stack(lit, 1).max(1).values)
        self.rule_names = list(rules)
        self.register_buffer('worlds', worlds)
        self.register_buffer('sat', torch.stack(sat, 1) if sat else torch.zeros(len(worlds), 0))  # [W,R]
        self.w = nn.Parameter(torch.full((len(rules),), float(init_w)))
        self.a = nn.Parameter(torch.ones(K))

    def forward(self, L):
        """L: [N, K] evidence log-odds. Returns the log-odds of query = 1."""
        s = (L * self.a) @ self.worlds.T + (self.sat @ self.w)  # [N,W]
        q = self.worlds[:, self.query].bool()
        return torch.logsumexp(s[:, q], 1) - torch.logsumexp(s[:, ~q], 1)

    def fit(self, L, y, iters=200, l2=1e-3):
        opt = torch.optim.LBFGS(self.parameters(), max_iter=iters, line_search_fn='strong_wolfe')
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(self(L), y.float()) + l2 * (self.w.pow(2).sum() + (self.a - 1).pow(2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        return {n: round(v, 3) for n, v in zip(self.rule_names, self.w.tolist())} | \
               {f'scale_{p}': round(v, 3) for p, v in zip(self.preds, self.a.tolist())}


# KnowGraph's LANL rules: R1 main => auth, R2 encg => main, R3 not encg => not main
KNOWGRAPH_RULES = {
    'R1_auth': [('main', False), ('auth', True)],
    'R2_encg_mal': [('encg', False), ('main', True)],
    'R3_encg_ben': [('encg', True), ('main', False)],
}
