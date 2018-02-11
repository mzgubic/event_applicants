import numpy as np
import pandas as pd
import datetime
# maffs
import scipy.optimize as spo
import scipy.integrate as integrate
from scipy.special import factorial
# plotting
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 20

class Prediction:

    def __init__(self, beta0):
        
        self.beta0 = beta0
        self.data = pd.read_csv('registrations.csv')
        self._clean_data()

        self.registration_start = '2018-02-01'
        self.registration_end = '2018-04-01'
        self._extract_info()

    def _clean_data(self):

        self.data = self.data[self.data.Name != 'Miha Zgubic']

    def _extract_info(self):

        # handle the registrations
        self.data.index = pd.DatetimeIndex(self.data['Registration date'])
        self.regs = self.data.groupby(self.data.index.date).count()['ID']
        self.regs.index = pd.DatetimeIndex(self.regs.index)

        # reindex to fill no registration days
        self.last_day = max(max(self.regs.index), pd.to_datetime(datetime.date.today()))
        idx = pd.date_range(self.registration_start, self.last_day)
        self.regs = self.regs.reindex(idx, fill_value=0)

        # compute cumulative registrations
        self.n = (pd.to_datetime(self.last_day) - pd.to_datetime(self.registration_start)).days + 1 
        self.m = (pd.to_datetime(self.registration_end) - pd.to_datetime(self.last_day)).days
        self.N = (pd.to_datetime(self.registration_end) - pd.to_datetime(self.registration_start)).days + 1
        print(self.n, self.m, self.N)

    def compute_predictions(self):

        self.alphas = self.regs.cumsum() + 1
        self.betas = pd.Series(np.arange(1, self.n+1), self.alphas.index) + self.beta0
        self.lambda_best = (self.alphas - 1) / self.betas

        self._compute_lower_and_upper()

        self.pred_mean = (self.alphas - 1) + self.lambda_best * (self.N - self.betas)
        self.pred_1_up = (self.alphas - 1) + self.lambda_1_up * (self.N - self.betas)
        self.pred_1_down = (self.alphas - 1) + self.lambda_1_down * (self.N - self.betas)

    def _compute_lower_and_upper(self):

        self.lambda_1_up = pd.Series(np.arange(1.0, self.n+1), self.alphas.index)
        self.lambda_1_down = pd.Series(np.arange(1.0, self.n+1), self.alphas.index)

        for day in self.alphas.index:

            lambda_best = self.lambda_best.get(day)
            alpha = self.alphas.get(day)
            beta = self.betas.get(day)

            # integral from 0 to lambda best can sometimes be less than 0.34
            try:
                down_1 = spo.bisect(self._f_lower, 0, lambda_best, args=(lambda_best, alpha, beta))
            except ValueError:
                down_1 = 0

            # always bigger than 0.34
            try:
                up_1 = spo.bisect(self._f_upper, lambda_best, 10, args=(lambda_best, alpha, beta))
            except ValueError:
                up_1 = 5

            self.lambda_1_down.set_value(day, down_1)
            self.lambda_1_up.set_value(day, up_1)

    def _f_lower(self, lower, upper, alpha, beta):
        result = integrate.quad(lambda x: beta**alpha / factorial(alpha-1) * x**(alpha-1) * np.e**(-beta*x), lower, upper)
        return result[0] - 0.34

    def _f_upper(self, upper, lower, alpha, beta):
        result = integrate.quad(lambda x: beta**alpha / factorial(alpha-1) * x**(alpha-1) * np.e**(-beta*x), lower, upper)
        return result[0] - 0.34

    def plot_predictions(self, upto=None):

        # decide what is the last day to plot prediction for
        if type(upto) != pd.Timestamp or upto > self.last_day:
            upto = self.last_day

        # prepare the plot
        fig, ax = plt.subplots(2, figsize=(15, 20))

        # decide how far in dates to plot
        date = self.regs.index
        endpoint = date.get_loc(upto)+1
        date = date[:endpoint]

        regs = self.regs[:endpoint]
        cumsum = self.regs.cumsum()[:endpoint]
        pred_mean = self.pred_mean[:endpoint]
        pred_1_up = self.pred_1_up[:endpoint]
        pred_1_down = self.pred_1_down[:endpoint]

        # main things
        ax[0].scatter(date, regs, c='k', marker='x', label='daily registrations')
        ax[0].plot(date, cumsum, c='k', label='cumulative registrations')
        ax[0].plot(date, pred_mean, c='C0', alpha=1.0, label='prediction for total registrations')
        ax[0].plot(date, [62 for i in date], c='C1', label='school capacity')
        ax[0].fill_between(date, pred_1_down, pred_1_up, label='1 sigma uncertainty band', alpha=0.5,
        facecolor='C0')

        xs = np.linspace(0, 5, 100)
        alpha = self.alphas.get(upto)
        beta = self.alphas.get(upto)
        pdf = beta**alpha / factorial(alpha-1) * xs**(alpha-1) * np.e**(-beta*xs)
        ax[1].plot(xs, pdf)

        # legend and cosmetics
        ax[0].legend(loc='best')
        ax[0].set_xlim(pd.to_datetime(self.registration_start), pd.to_datetime(self.registration_end))
        ax[0].set_title('Prediction for total number of MLHEP applications')
        ax[1].set_title('Posterior pdf for mean number of applicants per day')

        plt.sca(ax[0])
        plt.xticks(rotation=30)

        plt.savefig('{date}.pdf'.format(date=upto))
        #plt.show()        


def main():
    print('im so bayesian')
    
    shaman = Prediction(0.01)
    shaman.compute_predictions()
    for date in pd.date_range('2018-02-01', '2018-02-11'):
        shaman.plot_predictions(date)


if __name__ == '__main__':
    main()
