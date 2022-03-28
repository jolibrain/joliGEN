class IterCalculator:
    def __init__(self, loss_names):
        self.loss_names = loss_names

        for loss_name in self.loss_names:
            setattr(self, "loss_" + loss_name, 0)
            setattr(self, "loss_" + loss_name + "_cur", 0)

    def compute_last_step(self, loss_names):
        for loss_name in loss_names:
            setattr(
                self, "loss_" + loss_name, getattr(self, "loss_" + loss_name + "_cur")
            )
            setattr(self, "loss_" + loss_name + "_cur", 0)

    def compute_step(self, loss_name, value):
        old_value = getattr(self, "loss_" + loss_name + "_cur")
        setattr(self, "loss_" + loss_name + "_cur", old_value + value)
