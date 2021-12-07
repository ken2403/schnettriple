from schnetpack.train.hooks import Hook


__all__ = ["NanStoppingHook"]


class NanStoppingHook(Hook):
    def on_batch_end(self, trainer, train_batch, result, loss):
        if loss.isnan():
            trainer._stop = True
