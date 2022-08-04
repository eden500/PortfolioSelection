import pandas as pd
import yfinance as yf
from portfolio import Portfolio
import numpy as np
import os.path


START_DATE = '2022-04-01'
END_TRAIN_DATE = '2022-05-30'
END_TEST_DATE = '2022-06-30'

date1 = {'START_DATE': '2022-04-01'
    , 'END_TRAIN_DATE': '2022-05-30'
    , 'END_TEST_DATE': '2022-06-30'
}

date2 = {'START_DATE': '2022-03-01'
    , 'END_TRAIN_DATE': '2022-04-30'
    , 'END_TEST_DATE': '2022-05-30'
}

date3 = {'START_DATE': '2022-02-01'
    , 'END_TRAIN_DATE': '2022-03-30'
    , 'END_TEST_DATE': '2022-04-30'
}

dates = [date1, date2, date3]


def get_data():
    # if os.path.isfile('data.pd'):
    if False:
        return pd.read_pickle('data.pd')
    else:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        data = yf.download(tickers, START_DATE, END_TEST_DATE)
        data.to_pickle('data.pd')
    return data


def test_portfolio():
    for date in dates:
        globals().update(date)
        full_train = get_data()
        returns = []
        for k in [1, 5, 10]:
            for weight in [10, 1000]:
                for days in [31, 10, 5]:
                    strategy = Portfolio(k, weight, days)
                    for test_date in pd.date_range(END_TRAIN_DATE, END_TEST_DATE):
                        if test_date not in full_train.index:
                            continue
                        train = full_train[full_train.index < test_date]
                        cur_portfolio = strategy.get_portfolio(train)
                        if not np.isclose(cur_portfolio.sum(), 1):
                            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
                        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
                        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
                        test_data = test_data / prev_test_data - 1
                        cur_return = cur_portfolio @ test_data
                        returns.append({'date': test_date, 'return': cur_return})
                    returns = pd.DataFrame(returns).set_index('date')
                    mean_return, std_returns = float(returns.mean()), float(returns.std())
                    sharpe = mean_return / std_returns
                    print(sharpe)
                    print(f"k={k} w={weight} d = {days}")
                    returns = []

if __name__ == '__main__':
    test_portfolio()
