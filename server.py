from flask import Flask, render_template

app = Flask(__name__)

# Example endpoint: GET request

@app.route('/')
def dashboard():
    # Example data from API
    data = [
        {"title": "Card 1", "value": "123"},
        {"title": "Card 2", "value": "456"},
        {"title": "Card 3", "value": "789"},
        {"title": "Card 4", "value": "101"}
    ]
    return render_template('dashboard.html', cards=data)


if __name__ == '__main__':
    app.run(debug=True)