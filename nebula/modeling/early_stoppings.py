from nebula.commons import Logger

logger = Logger()


class EarlyStopping:
    def __init__(self, patience: int, metric_name: str = "f1", mode: str = "max"):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> bool:
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f" EarlyStopping counter: {self.counter}/{self.patience} "
                f"(best {self.metric_name}: {self.best_score:.4f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered! No improvement in {self.metric_name} "
                    f"for {self.patience} epochs."
                )
                logger.info(f"Best {self.metric_name}: {self.best_score:.4f}")
                return True

        return False
