import gradio as gr
import numpy as np
import pandas as pd
import json
import time
import argparse

from transformers import AutoModelForSequenceClassification

try:
	from utils.inference_utils import load_model_and_tokenizer, predict
	from utils.perf_utils import calc_acc, calc_prf
	from utils.reliability_utils import reliability_diagram
except ModuleNotFoundError as e:
	from seq_clf.utils.inference_utils import load_model_and_tokenizer, predict
	from seq_clf.utils.perf_utils import calc_acc, calc_prf
	from seq_clf.utils.reliability_utils import reliability_diagram
except Exception as e:
	raise e

def predict_with_text(
		batch_size,
		text,
		text_sep,
		progress = gr.Progress()
	):
	progress(0, desc="Starting...")
	sources = text.split(text_sep)
	predicted = predict(
		model,
		tokenizer,
		sources,
		device = device,
		batch_size = batch_size,
		seq_max_length = 512
	)
	y_pred = [p["label"] for p in predicted]
	y_probs = [json.dumps(["{:.4f}".format(x) for x in p["label_prob"]]) for p in predicted]

	processed_dict = {
		"source": sources,
		"predicted": y_pred,
		"probs": y_probs
	}
	df = pd.DataFrame.from_dict(processed_dict)
	return gr.Dataframe(df)

def make_clf_perf_eval_interface():
	batch_size = gr.Slider(8, 1024, value=16, label="Batch Size", info="between 8 ~ 1024")

	input_text = gr.Textbox(placeholder = "file_dir")
	input_sep = gr.Textbox(placeholder = "<<|SEP|>>")


	##
	inputs = [
		## Model
		batch_size,
		## Files
		input_text,
		input_sep,
	]
	interface = gr.Interface(fn=predict_with_text, inputs=inputs, outputs=["dataframe"])
	return interface

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str)
	parser.add_argument("--device", type=str)
	args = parser.parse_args()

	model_dir = args.model
	if args.device not in ["cpu", "mps", "cuda"]:
		raise ValueError(f"device should be one of cpu,cuda,mps")
	device = args.device
	model, tokenizer = load_model_and_tokenizer(model_dir, device = device, model_type = "classifier")

	interface = make_clf_perf_eval_interface()
	interface.launch()