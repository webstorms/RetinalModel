
class RetinalParameters:

    def __init__(self, n_in, rf_size, encoder_span, decoder_span, photo_noise, ganglion_noise, noise_type, mem_tc, dt):
        self.n_in = n_in
        self.rf_size = rf_size
        self.encoder_span = encoder_span
        self.decoder_span = decoder_span
        self.photo_noise = photo_noise
        self.ganglion_noise = ganglion_noise
        self.noise_type = noise_type
        self.mem_tc = mem_tc
        self.dt = dt

    @property
    def hyperparams(self):
        return {"n_in": self.n_in, "rf_size": self.rf_size, "encoder_span": self.encoder_span, "decoder_span": self.decoder_span, "photo_noise": self.photo_noise, "ganglion_noise": self.ganglion_noise, "noise_type": self.noise_type, "mem_tc": self.mem_tc, "dt": self.dt}
