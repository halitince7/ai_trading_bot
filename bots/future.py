from config import config_erkan
from binance.um_futures import UMFutures
import ta
import pandas as pd
from time import sleep
from binance.error import ClientError

client = UMFutures(key = config_erkan['key'], secret=config_erkan['secret'])

# 0.012 means +1.2%, 0.009 is -0.9%
tp = 0.009
sl = 0.009
volume = 10  # volume for one order (if its 10 and leverage is 10, then you put 1 usdt to one position)
leverage = 1
type = 'ISOLATED'  # type is 'ISOLATED' or 'CROSS'
qty = 5  # Amount of concurrent opened positions

# getting your futures balance in USDT
def get_balance_usdt():
    try:
        response = client.balance(recvWindow=6000)
        for elem in response:
            if elem['asset'] == 'USDT':
                return float(elem['balance'])

    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Getting all available symbols on the Futures ('BTCUSDT', 'ETHUSDT', ....)
def get_tickers_usdt():
    tickers = []
    resp = client.ticker_price()
    for elem in resp:
        if 'USDT' in elem['symbol']:
            tickers.append(elem['symbol'])
    return tickers



# Getting candles for the needed symbol, its a dataframe with 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'
def klines(symbol):
    try:
        resp = pd.DataFrame(client.klines(symbol, '1h'))
        resp = resp.iloc[:,:6]
        resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        resp = resp.set_index('Time')
        resp.index = pd.to_datetime(resp.index, unit = 'ms')
        resp = resp.astype(float)
        return resp
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


# Set leverage for the needed symbol. You need this bcz different symbols can have different leverage
def set_leverage(symbol, level):
    try:
        response = client.change_leverage(
            symbol=symbol, leverage=level, recvWindow=6000
        )
        print(response)
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )




# Price precision. BTC has 1, XRP has 4
def get_price_precision(symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['pricePrecision']


# Amount precision. BTC has 3, XRP has 1
def get_qty_precision(symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['quantityPrecision']


# Open new order with the last price, and set TP and SL:
def open_order(symbol, side):
    price = float(client.ticker_price(symbol)['price'])
    qty_precision = get_qty_precision(symbol)
    price_precision = get_price_precision(symbol)
    qty = round(volume/price, qty_precision)

    if side == 'buy':
        try:
            resp1 = client.new_order(symbol=symbol, 
                                     side='BUY', 
                                     type='MARKET', 
                                     quantity=qty)
            
            print(symbol, side, "placing order")
            print(resp1)
            sleep(2)

            sl_price = round(price - price*sl, price_precision)
            resp2 = client.new_order(symbol=symbol, 
                                     side='SELL', 
                                     type='STOP_MARKET', 
                                     quantity=qty, 
                                     timeInForce='GTC', 
                                     stopPrice=sl_price)
            print(resp2)
            sleep(2)

            tp_price = round(price + price * tp, price_precision)
            resp3 = client.new_order(symbol=symbol, 
                                     side='SELL', 
                                     type='TAKE_PROFIT_MARKET', 
                                     quantity=qty, 
                                     timeInForce='GTC',
                                     stopPrice=tp_price)
            print(resp3)
        except ClientError as error:
            print(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
    if side == 'sell':
        try:
            resp1 = client.new_order(symbol=symbol, 
                                     side='SELL', 
                                     type='MARKET', 
                                     quantity=qty)
            
            print(symbol, side, "placing order")

            #print(resp1)
            sleep(2)
            sl_price = round(price + price*sl, price_precision)
            resp2 = client.new_order(symbol=symbol, 
                                     side='BUY', 
                                     type='STOP_MARKET', 
                                     quantity=qty, 
                                     timeInForce='GTC', 
                                     stopPrice=sl_price)
            #print(resp2)
            sleep(2)
            tp_price = round(price - price * tp, price_precision)
            resp3 = client.new_order(symbol=symbol, 
                                     side='BUY', 
                                     type='TAKE_PROFIT_MARKET', 
                                     quantity=qty, 
                                     timeInForce='GTC',
                                     stopPrice=tp_price)
            print(resp3)
        except ClientError as error:
            print(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )


# Your current positions (returns the symbols list):
def get_pos():
    try:
        resp = client.get_position_risk()
        pos = []
        for elem in resp:
            if float(elem['positionAmt']) != 0:
                pos.append(elem['symbol'])
        return pos
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

def check_orders():
    try:
        response = client.get_orders(recvWindow=6000)
        sym = []
        for elem in response:
            sym.append(elem['symbol'])
        return sym
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

# Close open orders for the needed symbol. If one stop order is executed and another one is still there
def close_open_orders(symbol):
    try:
        response = client.cancel_open_orders(symbol=symbol, recvWindow=6000)
        print(response)
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )



def rsi_signal(symbol):
    kl = klines(symbol)
    rsi = ta.momentum.RSIIndicator(kl.Close).rsi()
    ema = ta.trend.ema_indicator(kl.Close, window=200)
    if rsi.iloc[-2] < 30 and rsi.iloc[-1] > 30:
        return 'up'
    if rsi.iloc[-2] > 70 and rsi.iloc[-1] < 70:
        return 'down'

    else:
        return 'none'



def macd_ema(symbol):
    kl = klines(symbol)
    macd = ta.trend.macd_diff(kl.Close)
    ema = ta.trend.ema_indicator(kl.Close, window=200)
    if macd.iloc[-3] < 0 and macd.iloc[-2] < 0 and macd.iloc[-1] > 0 and ema.iloc[-1] < kl.Close.iloc[-1]:
        return 'up'
    if macd.iloc[-3] > 0 and macd.iloc[-2] > 0 and macd.iloc[-1] < 0 and ema.iloc[-1] > kl.Close.iloc[-1]:
        return 'down'
    else:
        return 'none'



def main():
    symbol = ''
    symbols = get_tickers_usdt()

    while True:
        # we need to get balance to check if the connection is good, or you have all the needed permissions
        balance = get_balance_usdt()
        print(balance)
        sleep(1)
        if balance != None:
            print("My balance is: ", balance, " USDT")
            # getting position list:
            pos = []
            pos = get_pos()
            print(f'You have {len(pos)} opened positions:\n{pos}')
            # Getting order list
            ord = []
            ord = check_orders()
            # removing stop orders for closed positions
            for elem in ord:
                if not elem in pos:
                    close_open_orders(elem)

            if len(pos) < qty:
                for elem in symbols:
                    # Strategies (you can make your own with the TA library):

                    # signal = str_signal(elem)
                    #signal = rsi_signal(elem)
                    signal = macd_ema(elem)
                    if signal != 'none':
                        print(elem,signal)

                    # 'up' or 'down' signal, we place orders for symbols that arent in the opened positions and orders
                    # we also dont need USDTUSDC because its 1:1 (dont need to spend money for the commission)
                    if signal == 'up' and elem != 'USDCUSDT' and not elem in pos and not elem in ord and elem != symbol:
                        print('Found BUY signal for ', elem)
                        
                        client.change_margin_type(symbol=elem, marginType=type, recvWindow=6000)
                        sleep(1)
                        
                        client.change_leverage(symbol=symbol, leverage=leverage, recvWindow=6000)
                        sleep(1)
                        
                        print('Placing order for ', elem)
                        
                        open_order(elem, 'buy')
                        symbol = elem
                        pos = get_pos()
                        sleep(1)
                        ord = check_orders()
                        sleep(10)

                    if signal == 'down' and elem != 'USDCUSDT' and not elem in pos and not elem in ord and elem != symbol:
                        print('Found SELL signal for ', elem)
                        
                        client.change_margin_type(symbol=elem, marginType=type, recvWindow=6000)
                        sleep(1)
                        
                        client.change_leverage(symbol=symbol, leverage=leverage, recvWindow=6000)
                        sleep(1)
                        
                        print('Placing order for ', elem)
                        open_order(elem, 'sell')
                        symbol = elem
                        pos = get_pos()
                        sleep(1)
                        ord = check_orders()
                        sleep(10)
        print('Waiting 3 min')
        sleep(60)


if __name__ == "__main__":
    main()





