from schnetpack.train.hooks import Hook


__all__ = ["NanStoppingHook"]


class NanStopError(Exception):
    pass


class NanStoppingHook(Hook):
    def on_batch_end(self, trainer, train_batch, result, loss):
        if loss.isnan():
            trainer._stop = True
            # raise NanStopError("The value of loss has become nan! Stop training.")
