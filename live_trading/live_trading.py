# Step 1: Move to project root
%cd c:/Users/moham/OneDrive/ml_bot_trading

import warnings
warnings.filterwarnings("ignore")
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import time
import logging
import requests
import telegram
from telegram.ext import Updater, CommandHandler
import joblib


# Setup logging
logging.basicConfig(filename='trading_app1.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

def log_and_print(message, is_error=False):
    if is_error:
        logging.error(message)
    else:
        logging.info(message)
    print(message)

# Update the login credentials and server information accordingly
name = 52076807
key = 'ST7s$n9cFxEG38'
serv = 'ICMarketsSC-Demo'


# Global variables
SYMBOL = "EURUSD"
LOT_SIZE = 0.01
TIMEFRAME = mt5.TIMEFRAME_D1
N_BARS = 50000
MAGIC_NUMBER = 234003
SLEEP_TIME = 86400  # 4 hours in seconds
COMMENT_ML = "RFFV-D"


# Load sensitive data from environment variables for security
TELEGRAM_BOT_TOKEN = "6297037845:AAFt1Dv3xWClTnef3k47xQbObKRkhRz-Zow"
TELEGRAM_CHAT_ID = "161645988"
# Initialize the Telegram bot
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

def send_telegram_message(message, additional_info=None):
    """Function to send a message to the Telegram bot, with optional additional information."""
    full_message = message
    if additional_info:
        full_message += "\n" + additional_info  # Append additional information to the main message
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": full_message}
        response = requests.post(url, data=data)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        print("Message sent successfully!")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to connect to Telegram API: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")



def select_features_rf_reg(X, y, estimator, max_features=20):
    selector = SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=max_features).fit(X, y)
    X_transformed = selector.transform(X)
    selected_features_mask = selector.get_support()
    return X_transformed, selected_features_mask


class TradingApp:
    def __init__(self, symbol, lot_size, magic_number):
        self.symbol = symbol
        self.lot_size = lot_size
        self.magic_number = magic_number
        self.pipeline = None  # We'll store the loaded pipeline here
        self.last_retrain_time = None

    def get_data(self, symbol, n, timeframe):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame.set_index('time', inplace=True)
        return rates_frame

    def add_all_ta_features(self, df):
        """Add technical analysis features to the DataFrame."""
        df = ta.add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True
        )
        return df

    def load_pipeline(self, pipeline_path):
        """
        Loads a pre-trained pipeline (scaler + model + possibly feature selection)
        from disk, e.g. best_rf_pipeline.pkl
        """
        self.pipeline = joblib.load(pipeline_path)
        logging.info(f"Loaded pipeline from {pipeline_path}")

    def ml_signal_generation(self, symbol, n_bars, timeframe):
        """
        Generate buy/sell signals using the loaded pipeline.
        Make sure the pipeline expects the same features as we create below.
        """
        if self.pipeline is None:
            logging.error("No pipeline loaded. Call load_pipeline(...) first.")
            return False, False, True, True

        # 1) Fetch new data
        df = self.get_data(symbol, n_bars, timeframe)
        # 2) Add TA features (if your pipeline doesn't handle feature eng, do it here)
        df = self.add_all_ta_features(df)
        df.fillna(method='ffill', inplace=True)

        # 3) Prepare the features (the pipeline will do scaling/selection if included)
        # The pipeline expects columns in the same order as training
        X_new = df  # or df[self.selected_columns] if needed

        # 4) Predict with the pipeline
        predictions = self.pipeline.predict(X_new)
        latest_pred = predictions[-1]  # get the most recent bar's prediction

        buy_signal = latest_pred > 0
        sell_signal = latest_pred < 0

        return buy_signal, sell_signal, not buy_signal, not sell_signal

    def calculate_future_returns(self, df):
        df["future_returns"] = df["close"].pct_change().shift(-1)
        return df.dropna()

 


    def orders(self, symbol, lot, is_buy=True, id_position=None, sl=None, tp=None):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            log_and_print(f"Symbol {symbol} not found, can't place order.", is_error=True)
            return "Symbol not found"

        # Make sure symbol is selected/visible
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                log_and_print(f"Failed to select symbol {symbol}", is_error=True)
                return "Symbol not visible or could not be selected."

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            log_and_print(f"Could not get tick info for {symbol}.", is_error=True)
            return "Tick info unavailable"

        # Check for valid bid/ask
        if tick_info.bid <= 0 or tick_info.ask <= 0:
            log_and_print(f"Zero or invalid bid/ask for {symbol}: bid={tick_info.bid}, ask={tick_info.ask}", is_error=True)
            return "Invalid prices"

        # ----------- LOT SIZE VALIDATION -----------
        lot = max(lot, symbol_info.volume_min)
        step = symbol_info.volume_step
        if step > 0:
            remainder = lot % step
            if remainder != 0:
                lot = lot - remainder + step
        if lot > symbol_info.volume_max:
            lot = symbol_info.volume_max

        log_and_print(f"Adjusted lot size to {lot} (min={symbol_info.volume_min}, step={symbol_info.volume_step}, max={symbol_info.volume_max})")

        # ----------- FORCE ORDER_FILLING_IOC -----------
        filling_mode = 1  # ORDER_FILLING_IOC

        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        order_price = tick_info.ask if is_buy else tick_info.bid
        deviation = 20

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": COMMENT_ML,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,  # ✅ Ensure ORDER_FILLING_IOC (1)
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        if id_position is not None:
            request["position"] = id_position

        log_and_print(f"Sending order request: {request}")
        result = mt5.order_send(request)

        order_type_str = "BUY" if is_buy else "SELL"
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_message = f"Order failed for {symbol}"
            if result:
                error_message += f", retcode={result.retcode}, comment={result.comment}"
            additional_info = (
                f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Order Type: {order_type_str}\n"
                f"Lot Size: {lot}\n"
                f"SL: {sl if sl else 'None'}\n"
                f"TP: {tp if tp else 'None'}\n"
                f"Comment: {COMMENT_ML}\n"
                f"Request: {request}\n"
                f"Result: {result}"
            )
            send_telegram_message(f"🚨 {error_message}", additional_info)
            log_and_print(f"Order failed details: {additional_info}", is_error=True)
        else:
            success_message = f"Order successful for {symbol}, comment={result.comment}"
            additional_info = (
                f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Order Type: {order_type_str}\n"
                f"Lot Size: {lot}\n"
                f"SL: {sl if sl else 'None'}\n"
                f"TP: {tp if tp else 'None'}\n"
                f"Comment: {COMMENT_ML}"
            )
            send_telegram_message(f"✅ {success_message}", additional_info)
            log_and_print(success_message)





    def get_positions_by_magic(self, symbol, magic_number):
        """Retrieve positions for a specific symbol and magic number."""
        all_positions = mt5.positions_get(symbol=symbol)
        if not all_positions:
            log_and_print("No positions found.", is_error=False)
            return []
        return [pos for pos in all_positions if pos.magic == magic_number]



    def run_strategy(self, symbol, lot, buy_signal, sell_signal):
        log_and_print("------------------------------------------------------------------")
        log_and_print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, SYMBOL: {symbol}, BUY SIGNAL: {buy_signal}, SELL SIGNAL: {sell_signal}")

        # Retrieve positions based on the magic number to manage trades specific to this instance
        positions = self.get_positions_by_magic(symbol, self.magic_number)
        has_buy = any(pos.type == mt5.POSITION_TYPE_BUY for pos in positions)
        has_sell = any(pos.type == mt5.POSITION_TYPE_SELL for pos in positions)

        # Decision making based on current signals and existing positions
        if buy_signal and not has_buy:
            if has_sell:
                log_and_print("Existing sell positions found. Attempting to close...")
                if self.close_position(symbol, is_buy=True):  # Close all sell positions before opening a buy
                    log_and_print("Sell positions closed. Placing new buy order.")
                    self.orders(symbol, lot, is_buy=True)
                else:
                    log_and_print("Failed to close sell positions.")
            else:
                self.orders(symbol, lot, is_buy=True)
        elif sell_signal and not has_sell:
            if has_buy:
                log_and_print("Existing buy positions found. Attempting to close...")
                if self.close_position(symbol, is_buy=False):  # Close all buy positions before opening a sell
                    log_and_print("Buy positions closed. Placing new sell order.")
                    self.orders(symbol, lot, is_buy=False)
                else:
                    log_and_print("Failed to close buy positions.")
            else:
                self.orders(symbol, lot, is_buy=False)
        else:
            log_and_print("Appropriate position already exists or no signal to act on.")



    def close_position(self, symbol, is_buy):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            log_and_print(f"No positions to close for symbol: {symbol}")
            return False

        initial_balance = mt5.account_info().balance  # Get initial account balance
        closed_any = False

        for position in positions:
            # Ensure we only close positions of the opposite type related to the current magic number
            if position.magic == self.magic_number and ((is_buy and position.type == mt5.POSITION_TYPE_SELL) or (not is_buy and position.type == mt5.POSITION_TYPE_BUY)):
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                    "position": position.ticket,
                    "deviation": 20,
                    "magic": self.magic_number,  # Ensure to use the right magic number
                    "comment": COMMENT_ML,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(close_request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    error_message = f"Failed to close position {position.ticket} for {symbol}: {result.retcode}"
                    log_and_print(error_message, is_error=True)
                    send_telegram_message(f"🚨 {error_message}")
                else:
                    log_and_print(f"Successfully closed position {position.ticket} for {symbol}")
                    closed_any = True

        if closed_any:
            final_balance = mt5.account_info().balance  # Get final account balance
            profit = final_balance - initial_balance  # Calculate profit
            success_message = f"Closed positions successfully, Profit: {profit}"
            log_and_print(success_message)
            send_telegram_message(f"✅ {success_message}")
            return True

        return False


    def check_and_execute_trades(self):
        mt5.symbol_select(self.symbol, True)
        buy, sell, _, _ = self.ml_signal_generation(self.symbol)
        self.run_strategy(self.symbol, self.lot_size, buy, sell)
        mt5.symbol_select(self.symbol, False)
        log_and_print("Waiting for new signals...")



def is_market_open():
    """ Check if the current time is within the typical Forex trading session, adjusted for CET/CEST """
    current_time_utc = datetime.utcnow()
    # Adjust for Central European Time (UTC+1) or Central European Summer Time (UTC+2)
    current_time_cet = current_time_utc + timedelta(hours=2) if time.localtime().tm_isdst else current_time_utc + timedelta(hours=1)
    
    # Market closes at Friday 10:00 PM CET and opens at Sunday 11:00 PM CET
    if current_time_cet.weekday() == 4 and current_time_cet.hour >= 22:  # Friday after 10 PM CET
        return False
    elif current_time_cet.weekday() == 6 and current_time_cet.hour < 23:  # Sunday before 11 PM CET
        return False
    elif current_time_cet.weekday() == 5:  # All day Saturday
        return False
    return True


if __name__ == "__main__":
    try:
        if not mt5.initialize(login=name, server=serv, password=key):
            log_and_print("Failed to initialize MetaTrader 5", is_error=True)
            exit()

        app = TradingApp(symbol=SYMBOL, lot_size=LOT_SIZE, magic_number=MAGIC_NUMBER)

        # 1) Instead of app.train_model(...), we load the saved pipeline
        pipeline_path = "saved_models/final_production_pipeline.pkl"
        app.load_pipeline(pipeline_path)
        log_and_print(f"Loaded final pipeline for {app.symbol}")

        while True:
            log_and_print("Checking market status...")
            if is_market_open():
                log_and_print("Market is open. Executing trades...")

                # 2) Generate signals using the loaded pipeline
                buy_signal, sell_signal, _, _ = app.ml_signal_generation(
                    symbol=app.symbol,
                    n_bars=N_BARS,
                    timeframe=TIMEFRAME
                )
                # 3) Run strategy
                app.run_strategy(app.symbol, app.lot_size, buy_signal, sell_signal)

                # (Optional) Periodic re-load or re-train is not needed if we rely on a static pipeline
            else:
                log_and_print("Market is closed. No actions performed.")
            time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        log_and_print("Shutdown signal received.")
        send_telegram_message("🛑 Trading application has been manually stopped.")
    except Exception as e:
        error_message = f"An error occurred: {e}"
        log_and_print(error_message, is_error=True)
        send_telegram_message(f"🚨 {error_message}")
    finally:
        mt5.shutdown()
        log_and_print("MetaTrader 5 shutdown completed.")
        send_telegram_message("💤 MetaTrader 5 shutdown completed.")





