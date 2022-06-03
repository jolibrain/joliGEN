class DiscriminatorInfo:
    def __init__(
        self,
        name,
        loss_name_D,
        loss_name_G,
        loss_type,
        fake_name=None,
        real_name=None,
        compute_every=1,
    ):
        (
            self.name,
            self.loss_name_D,
            self.loss_name_G,
            self.loss_type,
            self.fake_name,
            self.real_name,
            self.compute_every,
        ) = (
            name,
            loss_name_D,
            loss_name_G,
            loss_type,
            fake_name,
            real_name,
            compute_every,
        )
