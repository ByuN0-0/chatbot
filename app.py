from flask import Flask, render_template, request, jsonify

from rag_manager import RAGManager
from logger import logger

app = Flask(__name__)
try:
    rag_manager = RAGManager()
except Exception as e:
    logger.exception("Failed to initialize RAGManager: %s", e)
    raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form['message']
        response = rag_manager.deep_research(user_message)
        return jsonify({'response': response})
    except Exception as e:
        logger.exception("Error processing /chat request: %s", e)
        return jsonify({'response': '오류가 발생했습니다.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
