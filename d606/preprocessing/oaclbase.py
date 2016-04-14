from sklearn.base import TransformerMixin
from d606.preprocessing.oacl import estimate_theta, estimate_theta_multiproc, clean_signal, clean_signal_multiproc
from multiprocessing import Queue, Process
from numpy import array, median


class OACL(TransformerMixin):

    def __init__(self, ranges=((3, 7), (7, 15)), m=11, decimal_precision=300, multi_run=False, num_classes=4,
                 trials=False):
        self.theta = None
        self.ranges = ranges
        self.m = m
        self.decimal_precision = decimal_precision
        self.multi_run = multi_run
        self.num_classes = num_classes
        self.trials = trials
        self.artifacts = []

    def get_params(self):
        params = (self.ranges, self.m, self.decimal_precision, self.trials, self.num_classes)
        return params

    def fit(self, x, y):
        if self.multi_run is False:
            self.theta = estimate_theta(x, self.get_params())
        else:
            thetas = []
            processes = []

            n_runs = len(x)
            input_queue = Queue(n_runs)
            output_queue = Queue(n_runs)

            for i, z in enumerate(x):
                input_queue.put((z, i))

            for x in range(0, n_runs):
                p = Process(target=estimate_theta_multiproc, args=(input_queue, output_queue, self.get_params()))
                p.start()
                processes.append(p)

            for x in range(0, n_runs):
                thetas.append(output_queue.get())

            for p in processes:
                p.join()

            sort = sorted(thetas, key=lambda theta: theta[2])

            self.artifacts = [x[1] for x in sort]
            thetas = [x[0] for x in sort]

            self.theta = self.generalize_thetas(thetas)

        return self

    def transform(self, x):

        if self.theta is None:
            raise RuntimeError("It is not possible to transform the data," +
                               " no theta value was found")

        cleaned_signal = []

        if self.multi_run is False:
            cleaned_signal = (clean_signal(x, self.theta, self.get_params()))
        else:
            cleaned_signals = []
            processes = []
            n_runs = len(x)
            input_queue = Queue(n_runs)
            output_queue = Queue(n_runs)

            if self.trials is True:
                for i, z in enumerate(x):
                    input_queue.put((z, self.artifacts[i], i))
            else:
                for i, z in enumerate(x):
                    input_queue.put((z, i))

            for x in range(0, n_runs):
                p = Process(target=clean_signal_multiproc, args=(input_queue, output_queue, self.theta,
                                                                 self.get_params()))
                p.start()
                processes.append(p)

            for x in range(0, n_runs):
                cleaned_signal.append(output_queue.get())

            for p in processes:
                p.join()

            cleaned_signal = [x[0] for x in sorted(cleaned_signal, key=lambda index: index[1])]
        return cleaned_signal

    def generalize_thetas(self, thetas):
        generalized_thetas = []
        t = []
        for x in zip(*thetas):
            t.append([median([z[0] for z in x])])
            t.append([median([z[1] for z in x])])
            generalized_thetas.append(array(t))
            t = []
        return generalized_thetas
