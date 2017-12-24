from flask import Flask, render_template, request
from flask import jsonify
import tensorflow as tf
import numpy as np
import seq2seq_wrapper
import data
import data_utils

app = Flask(__name__, static_url_path="/static")

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='startrek/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

@app.route('/message', methods=['POST'])
def reply():
    input_msg_ = request.form['msg']
    msg = str(input_msg_).lower()
    msg = data.filter_line(msg, "0123456789abcdefghijklmnopqrstuvwxyz ")
    msg_arr = msg.split(' ')
    message = data.zero_pad_line(msg_arr, metadata['w2idx'])
    output = model.predict(sess, np.array(np.array(message.T)))
    decoded = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ').split(' ')
    return jsonify({'text': ' '.join(decoded)})

@app.route("/")
def index():
    return render_template("index.html")


xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                ckpt_path='ckpt/',
                                emb_dim=emb_dim,
                                num_layers=3
                                )


sess = model.restore_last_session()

if (__name__ == "__main__"):
    app.run(port=5000)
