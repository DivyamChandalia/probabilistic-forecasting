import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt backend
import matplotlib.pyplot as plt

import math
import torch
import random
import math
from multiprocess import Pool
from tqdm import tqdm
import pickle

class Data:
    def __init__(self, tickers, split, len_input, len_output, stride):
        self.tickers = tickers
        self.completed_tickers = []
        self.split = split
        self.len_input = len_input
        self.len_output = len_output
        self.stride = stride
        self.num_features = 5
        self.num_tickers = len(tickers)
        self.mean = {}
        self.std = {}
        self.test_data = {} #{'symbol': [[data],[data]]}
        self.train_data = {}

        self.mode = "train" #mode can be "train" or "test" or "val"
        self.test_pairs = []
        self.train_pairs = []
        self.val_pairs = []

    def generate_features(self, Open, Close, Low, High, Volume, prev_vol):
        change_price = (Close - Open) / Open
        change_vol = (Volume - prev_vol) / prev_vol
        deviation_low = (High - Open) / Open
        deviation_high = (Low - Close) / Close
        deviation_centre = (Close - Open) / 2 - (High - Low) / 2
        deviation = (High - Low) / Low
        # return [change_price, change_vol, deviation_centre, deviation]
        return [Open, Close, Low, High, Volume]

    def preprocess(self, symbol):
        ticker = yf.Ticker(symbol)
        ticker_data = ticker.history(period="max",timeout=15)
        # window_size = 20
        # ticker_data = ticker_data.apply(lambda x: x.rolling(window=window_size).mean())
        temp_list = []
        ticker_dict = {symbol: []}
        prev_vol = ticker_data.iloc[0]['Volume']
        for index, row in ticker_data.iterrows():
            row = row[['Open','Close','High','Low','Volume']].apply(np.log) #debatable
            bad_data_present = row[['Open', 'Close', 'High', 'Low', 'Volume']].isna().any() or \
                               row[['Open', 'Close', 'High', 'Low', 'Volume']].isnull().any() or \
                               row[['Open', 'Close', 'High', 'Low', 'Volume']].isin([np.inf, -np.inf]).any()
            if not bad_data_present:
                temp_list.append(self.generate_features(row['Open'], row['Close'], row['Low'], row['High'], row['Volume'], prev_vol))
                prev_vol = row['Volume']
            elif ticker_data.index.get_loc(index) + 1 < len(ticker_data.index):
                prev_vol = ticker_data.iloc[ticker_data.index.get_loc(index) + 1]['Volume']
                if len(temp_list)>0:
                    ticker_dict[symbol].append(np.array(temp_list))
                    temp_list = []
        if len(temp_list)>0:
            ticker_dict[symbol].append(np.array(temp_list))

        return ticker_dict

    def len_ticker(self, ticker_data):
        len = 0
        for data in ticker_data:
            len = len + data.shape[0]
        return len
    # load_data
    def split_test(self, ticker_dict, symbol):

        ticker_len = self.len_ticker(ticker_dict[symbol])
        train_data = {}

        len_train = math.floor(ticker_len * self.split[0])
        len_val = math.floor(ticker_len * self.split[1])
        len_test = ticker_len - len_train - len_val
        lendiv = len(ticker_dict[symbol][-1])
        if lendiv < len_test:
            len_test = lendiv

        if len_test >= self.len_input + self.len_output :
            self.test_data[symbol] = []
            self.test_data[symbol].append(ticker_dict[symbol][-1][-len_test:])
            train_data[symbol] = ticker_dict[symbol][:-1]
            if len_test - lendiv >= self.len_input + self.len_output:
                train_data[symbol].append(ticker_dict[symbol][-1][:len_test])
        else:
            train_data[symbol] = ticker_dict[symbol]

        return train_data

    def normalise(self, ticker_dict, symbol):

        array_list = ticker_dict[symbol]
        long_array = np.concatenate(array_list, axis=0)

        # Calculate the mean and standard deviation of the long array along each column
        mean = np.mean(long_array, axis=0)
        std = np.std(long_array, axis=0)

        # Normalize the long array
        normalized_array = (long_array - mean) / std

        # Split the normalized array back into the original list of arrays
        ticker_dict[symbol] = np.split(normalized_array, np.cumsum([arr.shape[0] for arr in array_list])[:-1])

        self.mean[symbol] = mean
        self.std[symbol] = std
        return ticker_dict

    def make_pairs(self, ticker_dict, symbol):
        ticker_length = len(self.tickers)
        pairs = []

        for i, data in enumerate(ticker_dict[symbol]):
            len_data = len(data)

            if len_data > self.len_input + self.len_output:
                total_iterations = len_data - self.len_input - self.len_output

                for j in range(0, total_iterations, self.stride):
                    #add ticker <SOS>
                    ticker_data = torch.zeros(ticker_length)
                    ticker_data[self.tickers.index(symbol)] = 1

                    src = torch.zeros(self.len_input + self.len_output, self.num_features)

                    for k in range(j, self.len_input + self.len_output + j):
                        src[k - j] = torch.tensor(data[k])

                    pairs.append((ticker_data, src))
        return pairs

    def split_val(self,pairs):

        random.shuffle(pairs)
        len_pairs = len(pairs)
        train_proportion = self.split[0] / (self.split[0] + self.split[1])
        train_size = int(len_pairs * train_proportion)
        val_size = len_pairs - train_size

        train_data = pairs[:train_size]
        val_data = pairs[train_size:]

        self.val_pairs = self.val_pairs + val_data

        return train_data

    def norm_pairs(self, pairs):
        norm = []
        for sos,src in pairs:
            input = src[:self.len_input]
            min, _ = torch.min(input, axis=0)
            max, _ = torch.max(input, axis=0)
            norm.append([sos,(src - min) / (max - min)])
        return norm

    def load_normie(self):
        progress_bar = tqdm(self.tickers, desc='Loading Data')
        for symbol in progress_bar:
            if symbol in self.completed_tickers:
                continue
            progress_bar.set_description(f'Loading Data for {symbol}')
            ticker_dict = self.preprocess(symbol)
            # ticker_dict = self.normalise(ticker_dict, symbol)
            train_dict = self.split_test(ticker_dict, symbol)
            self.train_data[symbol] = train_dict[symbol]

            pairs = self.make_pairs(ticker_dict, symbol)
            pairs = self.norm_pairs(pairs)
            self.train_pairs = self.train_pairs + self.split_val(pairs)
            self.completed_tickers.append(symbol)


        for symbol, value in tqdm(self.test_data.items(), desc='Loading Test Data'):
            self.test_pairs = self.test_pairs + self.norm_pairs(self.make_pairs(self.test_data, symbol))

    def load(self):
        progress_bar = tqdm(self.tickers, desc='Loading Data')
        
        with Pool() as pool:
            for _ in pool.imap_unordered(self.load_ticker, self.tickers):
                progress_bar.update()

        for symbol, value in tqdm(self.test_data.items(), desc='Loading Test Data'):
            self.test_pairs = self.test_pairs + self.norm_pairs(self.make_pairs(self.test_data, symbol))

    
    def load_ticker(self, symbol):
        if symbol in self.completed_tickers:
                return
        # progress_bar.set_description(f'Loading Data for {symbol}')
        ticker_dict = self.preprocess(symbol)
        # ticker_dict = self.normalise(ticker_dict, symbol)
        train_dict = self.split_test(ticker_dict, symbol)
        self.train_data[symbol] = train_dict[symbol]

        pairs = self.make_pairs(ticker_dict, symbol)
        pairs = self.norm_pairs(pairs)
        self.train_pairs = self.train_pairs + self.split_val(pairs)
        self.completed_tickers.append(symbol)
        return

    def plot_features(self):
        fig, axs = plt.subplots(self.num_features, 1, figsize=(12, 8 * self.num_features), sharex=True)

        for symbol, data_list in self.test_data.items():
            # Concatenate the data from all the lists for the same symbol
            ticker_data = np.concatenate(data_list, axis=0)

            for i in range(self.num_features):
                # feature_name = ['Change Price', 'Change Volume', 'Deviation Centre', 'Deviation'][i]
                axs[i].plot(ticker_data[:, i], label=symbol)
                axs[i].set_ylabel(i)

        axs[-1].set_xlabel('Time')
        axs[0].legend()
        plt.tight_layout()
        plt.show()

    def plot_pair(self, index, fig):
        if index>len(self.train_pairs):
            print("index doesnt exist")
            return
        input_data = self.train_pairs[index][1][:self.len_input]
        output_data = self.train_pairs[index][1][self.len_input:]

        # Convert tensors to numpy arrays for plotting
        input_data_np = input_data.cpu().numpy()
        output_data_np = output_data.cpu().numpy()

        # Plot each feature of the input data and its corresponding output in separate plots
        num_features = input_data.shape[1]
        for i in range(num_features):
            fig.clear()
            plt.plot(input_data_np[:, i], label=f'Input Feature {i + 1}', color='blue')
            plt.plot(range(len(input_data_np), len(input_data_np) + len(output_data_np)), output_data_np[:, i],
                     label=f'Output Feature {i + 1}', linestyle='dashed', color='red')
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            # plt.show()
            plt.pause(1)
             
            

    def __len__(self):
        if self.mode == "train":
            return len(self.train_pairs)
        elif self.mode == "val":
            return len(self.val_pairs)
        elif self.mode == "test":
            return len(self.test_pairs)
        else:
            return len(self.train_pairs)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.train_pairs[index][1], self.train_pairs[index][0]
        elif self.mode == "val":
            return self.val_pairs[index][1], self.val_pairs[index][0]
        elif self.mode == "test":
            return self.test_pairs[index][1], self.test_pairs[index][0]
        else:
            return self.train_pairs[index][1], self.train_pairs[index][0]

def plot_losses(train_losses, val_losses, fig):
    fig.clear()
    #plot losses
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.pause(0.5)
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    # plt.show()

def calculate_probability(k, epoch):
    v = k / (k + torch.exp(torch.tensor(epoch / k)))
    return v

def flip_from_probability(p, batch_size, num_features):
    # Generate random numbers from a uniform distribution [0, 1]
    random_numbers = torch.rand(batch_size)
    # Create a boolean tensor where elements are True if the random number is less than p
    boolean_tensor = random_numbers < p

    boolean_tensor = boolean_tensor.unsqueeze(1).expand(-1, num_features)
    return boolean_tensor

def flip(p):
    return random.random() < p

def save_data(data, path):
    with open(path + "data.pickle", "wb") as file:
        pickle.dump(data, file)

def load_data(path):
    with open(path + "data.pickle", 'rb') as file:
        data = pickle.load(file)
    return data