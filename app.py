##
# Uses LangChain, LlamaIndex and UnstructuredReader loader to prepare the dataset for the vector
# Load an unstructured local file and vector it

## Dependecies:
# pip install flask
# pip install llama_index
# pip install langchain


from flask import Flask, render_template, request
from pathlib import Path
import os

from llama_index import GPTSimpleVectorIndex, download_loader, LLMPredictor, PromptHelper
from langchain import OpenAI


# set OpenAI API
# os.environ["OPENAI_API_KEY"] = ''
# in case it is already defined on windows path variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


## context data config ##
def load_data():
    UnstructuredReader = download_loader("UnstructuredReader")
    loader = UnstructuredReader()
    documents = loader.load_data(file=Path('./content/empresas.txt'))
    return documents


## AI config ##
def get_llm_predictor():
    # define AI model
    model = "text-davinci-003"
    # define creativity in the response
    creativity = 0.5
    # number of completions to generate
    completions = 1
    # set number of output tokens
    num_outputs = 600
    # params for LLM (Large Language Model)
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=creativity,
            model_name=model,
            max_tokens=num_outputs,
            n=completions
        )
    )
    return llm_predictor


# method to generate response querying the indexed data
def generate_response(prompt, index):
    response = index.query(prompt, response_mode="compact")
    return response.response


# array to store conversations
# define initial role
conversation = ["You are a virtual assistant and you speak portuguese."]

# load indexed data
documents = load_data()
index = GPTSimpleVectorIndex(documents, llm_predictor=get_llm_predictor())

app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    # load indexed data
    index = GPTSimpleVectorIndex(documents, llm_predictor=get_llm_predictor())

    user_input = request.args.get("msg") + '\n'
    response = ''
    if user_input:
        conversation.append(f"{user_input}")

        # get conversation history
        prompt = "\n".join(conversation[-3:])

        # generate AI response based on indexed data
        response = generate_response(prompt, index)

        # add AI response to conversation
        conversation.append(f"{response}")

    return response if response else "Sorry, I didn't understand that."


if __name__ == "__main__":
    app.run()
