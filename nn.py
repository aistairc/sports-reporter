import dynet as dy


class Linear:
    def __init__(self, input_size, output_size, model, bias=True, initial_weight=None):
        self.input_size, self.output_size = input_size, output_size
        self.W = model.add_parameters((output_size, input_size)) if initial_weight is None else initial_weight
        self.b = model.add_parameters((output_size,)) if bias else None

    def __call__(self, inputs):
        out = self.W * inputs
        if self.b is not None:
            out += self.b
        return out


class BiLinear:
    def __init__(self, input_size, query_size, model):
        self.W = model.add_parameters((input_size, query_size))

    def __call__(self, inputs, query):
        outs = dy.transpose(inputs) * (self.W * query)
        d, _ = outs.dim()
        return dy.transpose(outs) if len(d) > 1 else outs


class Sequential:
    def __init__(self, *args):
        self.layers = args

    def __call__(self, inputs):
        hidden = self.layers[0](inputs)
        for layer in self.layers[1:]:
            hidden = layer(hidden)
        return hidden


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, model, num_layers=1, nonlinear=dy.rectify):
        args = []
        for i in range(num_layers):
            args.extend([Linear(hidden_dim if i else input_dim, hidden_dim, model), nonlinear])
        args.append(Linear(hidden_dim, output_dim, model))
        self.layers = Sequential(*args)

    def __call__(self, inputs):
        return self.layers(inputs)
