import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import io
import base64
import yfinance as yf
from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

def get_stock_data(symbol, period="2mo"):
    print(f"Fetching stock data for {symbol} over {period}...")  
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)  

    if df.empty:
        print(f"No data available for {symbol}.")
        return None

    df = df.reset_index()  
    return df

def generate_plot(df):
    plt.figure(figsize=(10, 4))
    df["Date"] = pd.to_datetime(df["Date"])

    plt.plot(df["Date"], df["Close"], marker="o", linestyle="-", color="b", label="Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Stock Price Trend")
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator()) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")  
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()
    return plot_url

def predict_next_day_price(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days 
    X = df["Days"].values.reshape(-1, 1)
    y = df["Close"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[X[-1][0] + 1]])  
    predicted_price = model.predict(next_day)[0][0]

    return round(predicted_price, 2)  

@app.route("/", methods=["GET", "POST"])
def index():
    stock_price = None
    stock_data = None
    plot_url = None
    error_message = None
    predicted_price = None
    selected_period = "2mo"  

    if request.method == "POST":
        stock_symbol = request.form.get("symbol", "").upper().strip()
        selected_period = request.form.get("period", "2mo")

        print("Received stock symbol:", stock_symbol, "for period:", selected_period) 

        if stock_symbol:
            stock_data = get_stock_data(stock_symbol, selected_period)

            if stock_data is not None:
                stock_price = stock_data["Close"].iloc[-1]  
                plot_url = generate_plot(stock_data)
                predicted_price = predict_next_day_price(stock_data)  
            else:
                error_message = f"No data available for {stock_symbol}."
        else:
            error_message = "Please enter a stock symbol."

    return render_template(
        "index.html", 
        stock_price=stock_price, 
        stock_data=stock_data, 
        plot_url=plot_url, 
        predicted_price=predicted_price,
        error_message=error_message,
        selected_period=selected_period
    )

@app.route('/stkname')
def stkname():
    return render_template('stkname.html')

if __name__ == "__main__":
    app.run(debug=True)
