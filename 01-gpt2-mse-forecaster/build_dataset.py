import os
from utils import Data, save_data
import torch

file_path = os.path.dirname(os.path.abspath(__file__)) + "/"
# Define the list of ticker symbols you want to retrieve data for
crypto = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD",
        "ADA-USD", "SOL-USD", "TRX-USD", "LTC-USD", "MATIC-USD",
        "DOT-USD", "SHIB-USD", "BCH-USD", "AVAX-USD",
        "XLM-USD", "TON11419-USD", "LINK-USD", "UNI7083-USD",
        'LEO-USD', 'ATOM-USD', 'XMR-USD', 'ETC-USD', 'OKB-USD',
        'FIL-USD', 'ICP-USD', 'WHBAR-USD', 'HBAR-USD',
        'LDO-USD', 'APT21794-USD',
        'ARB11841-USD', 'CRO-USD', 'VET-USD', 'QNT-USD',
        'NEAR-USD', 'MKR-USD', 'OP-USD', 'HEX-USD',
        'GRT6719-USD', 'AAVE-USD', 'ALGO-USD', "AXS-USD",
        "XDC-USD", "EGLD-USD", "STX4847-USD", "SAND-USD",
        "EOS-USD", "IMX10603-USD", "FRAX-USD", "THETA-USD",
        "XTZ-USD", "BSV-USD", "KAS-USD", "MANA-USD",
        "APE18876-USD", "SNX-USD", "FTM-USD", "INJ-USD",
        "RNDR-USD", "TNC5524-USD", "NEO-USD",
        "XEC-USD", "FLOW-USD", "KAVA-USD", "RPL-USD",
        "XRD-USD", "KCS-USD", "CFX-USD", "CHZ-USD", "CRV-USD",
        "COMP5692-USD", "GALA-USD", "KLAY-USD", "PEPE24478-USD",
        "GMX11857-USD", "ZEC-USD", "LUNC-USD",
        "FTT-USD", "HT-USD", "FXS-USD"]

indexes = ["^GSPC", "^DJI", "^IXIC", "^NYA", "^XAX", "^RUT", "^FTSE","^GDAXI", "^FCHI",
        "^STOXX50E", "^N100", "^BFX", "^N225","^HSI", "000001.SS", "399001.SZ", "^STI", "^AXJO",
        "^AORD", "^BSESN", "^JKSE", "^KLSE", "^NZ50", "^KS11", "^TWII", "^GSPTSE", "^BVSP", "^MXX",] #confirmed indexes

indian_stocks = ["TATACONSUM.NS", "ULTRACEMCO.NS", "CIPLA.NS", "NESTLEIND.NS", "TITAN.NS", "BHARTIARTL.NS", "ITC.NS",
        "BAJAJ-AUTO.NS", "RELIANCE.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "LTIM.NS", "HEROMOTOCO.NS",
        "KOTAKBANK.NS", "BAJFINANCE.NS", "INDUSINDBK.NS", "ADANIENT.NS", "TATASTEEL.NS", "APOLLOHOSP.NS",
        "LT.NS", "MARUTI.NS", "WIPRO.NS", "TCS.NS", "BRITANNIA.NS", "HINDALCO.NS", "HDFCLIFE.NS", "ONGC.NS", "NTPC.NS"]

international_stocks = ["SOFI", "PLTR", "NIO", "TSLA", "XPEV", "F", "AMD", "LCID", "AMC", "ROKU", "INTC", "RIVN", "SNAP",
                        "PBR", "AAPL", "T", "GDRX", "NVDA", "CCL", "AFRM", "APLS", "PFE", "QS", "FSR",
                        "ETRN", "META", "BAC", "GOOGL", "DIS", "MSFT", "JNJ", "AI", "BABA", "SIRI", "NU", "IONQ",
                        "DIDIY", "XOM", "DNA", "AAL", "NOK", "VALE", "GOOG", "ABEV", "CVNA", "VZ", "KVUE",
                        "AUR", "PYPL", "UPST", "DKNG", "NCLH", "PBR-A", "NEWR", "JD", "KDP", "BB", "IQ", "PCG", "LU", "GRAB",
                        "KEY", "WBD", "JOBY", "TAL", "CHPT", "ITUB", "CMCSA", "LYG", "ARCC", "CSX", "LYFT", "HOOD", "PDD",
                        "PARA", "SHOP", "U", "JBLU", "BILI", "FTCH", "C", "ET", "COIN", "TME", "TSM"]

ticker_symbols = international_stocks + crypto + indexes + indian_stocks
# ticker_symbols = ["BTC-USD","ETH-USD", "XRP-USD","BNB-USD","ADA-USD",
#                   "SOL-USD","DOGE-USD","TRX-USD","MATIC-USD","LTC-USD"]  # Example: Bitcoin, Ethereum, Litecoin against USD
# ticker_symbols = indian_stocks

num_features = int(5)
num_tickers = len(ticker_symbols)
train_split, val_split, test_split = 0.6, 0.2, 0.2 #must add upto 1

input_seq_length = 600  # Adjust this based on your requirements
output_seq_length = 200  # Adjust this based on your requirements
stride = 200 #stride for the pairs

data = Data(tickers = ticker_symbols, split = [0.6,0.2,0.2], len_input = input_seq_length, len_output = output_seq_length, stride = stride)
# data.completed_tickers.append("LDO-USD")
data.load_normie()
save_data(data,file_path)