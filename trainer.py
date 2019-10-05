import random
from tqdm import tqdm

import dynet as dy


class Trainer:
    def __init__(self, model, lr=2e-3, decay=0.99, batch_size=32):
        self.model = model
        self.lr = lr
        self.lr_decay = decay
        self.opt = dy.AmsgradTrainer(model.pc, self.lr)
        self.opt.set_clip_threshold(2.)

        self.batch_size = batch_size
        self.loss = 0.
        self.iter = 0

    def __repr__(self):
        return "Learning Rate: {}, LR Decay: {}, Batch size: {}, ".format(self.lr, self.lr_decay, self.batch_size)

    def fit_partial(self, instances):
        random.shuffle(instances)
        self.iter += 1

        losses = []
        dy.renew_cg()

        total_loss, total_size = 0., 0
        prog = tqdm(desc="Epoch {}".format(self.iter), ncols=80, total=len(instances) + 1)
        for i, ins in enumerate(instances, 1):
            losses.extend(list(self.model.loss(*ins)))
            if i % self.batch_size == 0:
                loss = dy.sum_batches(dy.concatenate_to_batch(losses))
                total_loss += loss.value()
                total_size += len(losses)
                prog.set_postfix(loss=loss.value()/len(losses))

                loss.backward()
                self.opt.update()
                dy.renew_cg()
                losses = []

            prog.update()

        if losses:
            loss = dy.sum_batches(dy.concatenate_to_batch(losses))
            total_loss += loss.value()
            total_size += len(losses)
            self.loss = total_loss / total_size
            prog.set_postfix(loss=self.loss)

            loss.backward()
            self.opt.update()
            dy.renew_cg()

            prog.update()

        self.opt.learning_rate *= self.lr_decay
        prog.close()
