from sklearn.base import TransformerMixin
from d606.preprocessing.oacl import estimate_theta, estimate_theta_multiproc, clean_signal
from multiprocessing import Queue, Process


class OACL(TransformerMixin):

    def __init__(self, ranges=((3, 7), (7, 15)), m=11, decimal_precision=300, multi_run=False, num_classes=4):
        self.theta = None
        self.ranges = ranges
        self.m = m
        self.decimal_precision = decimal_precision
        self.multi_run = multi_run
        self.num_classes = num_classes

    def get_params(self):
        params = (self.ranges, self.m, self.decimal_precision, self.num_classes)
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

            thetas = sorted(thetas, key=lambda theta: theta[1])

            self.theta = self.generalize_thetas(thetas)

        return self

    def transform(self, x):

        if self.theta is None:
            raise RuntimeError("It is not possible to transform the data," +
                               " no theta value was found")
            return

        cleaned_signal = []

        if self.multi_run is False:
            cleaned_signal = (clean_signal(x, self.theta, self.get_params()))
        else:
            for i, run in enumerate(x):
                print "Cleaning for run " + str(i)
                cleaned_signal.append(clean_signal(run, self.theta, self.get_params()))

        return cleaned_signal

    def generalize_thetas(self, thetas):
        print "Not done yet"
        print([t[0][0][0][0] for t in thetas])
        return thetas[0][0]
        # Find a good way to generalize over thetas for each channel for several runs
