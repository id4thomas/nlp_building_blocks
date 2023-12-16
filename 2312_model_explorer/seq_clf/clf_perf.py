import gradio as gr
import numpy as np
import pandas as pd
import json
import time
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

def predict_with_eval_file(
		model_dir, 
		device,
		batch_size,
		file_dir,
		file_obj,
		file_sep,
		src_col_name,
		label_col_name,
		num_labels,
		target_label,
		num_confidence_bins,
		progress = gr.Progress()
	):
	progress(0, desc="Starting...")

	## Load CLF Model
	# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	model, tokenizer = load_model_and_tokenizer(model_dir, device = device, model_type = "classifier")
	progress(0.5, desc="Loaded Model, Predicting ...")

	## Load Data
	print("FILE DIR",file_dir)
	sep = "," if file_sep=="csv" else "\t"
	if file_dir:
		file_target = file_dir
	elif file_obj:
		file_target = file_obj
	else:
		raise ValueError("File not provided")


	try:
		df = pd.read_csv(file_target, sep = sep)
	except Exception as e:
		print(f"ERR {e} - trying lineterminator")
		df = pd.read_csv(file_target, sep = sep, lineterminator='\n')
	print("Loaded DF",df.shape, df.columns)

	sources = df[src_col_name].values.tolist()
	y_true = df[label_col_name].values.tolist()
	predicted = predict(
		model,
		tokenizer,
		sources,
		device = device,
		batch_size = batch_size,
		seq_max_length = 512
	)
	y_pred = [p["label"] for p in predicted]
	target_probs = [p["label_prob"][int(target_label)] for p in predicted]

	df["predicted"] = y_pred
	df["target_probs"] = target_probs
	
	progress(0.8, desc="Prediction Complete, Evaluating ...")
	## Model Classification Performance
	if int(num_labels) >2:
		is_binary = False
	else:
		is_binary = True
	acc = calc_acc(y_true, y_pred)
	perf_dict = calc_prf(y_true, y_pred, pos_label = int(target_label), is_binary=is_binary)
	perf_dict["accuracy"] = acc

	## Calibration Performance
	fig = reliability_diagram(
		np.array(y_true),
		np.array([1]*len(y_true)),
		np.array(target_probs),
		num_bins = int(num_confidence_bins),
		draw_ece = True,
		draw_bin_importance = "alpha",
		draw_averages = True,
		title = "Reliability Diagram",
		figsize=(8,8),
		dpi=100,
		return_fig=True
	)

	progress(0.8, desc="Complete")
	return gr.JSON(perf_dict), fig, gr.Dataframe(df)

def make_clf_perf_eval_interface():
	model_dir = gr.Textbox(placeholder = "model_dir")
	device_type = gr.Radio(["cpu", "cuda", "mps"], label="Device Type", info="Accelerator device type")
	batch_size = gr.Slider(8, 1024, value=16, label="Batch Size", info="between 8 ~ 1024")

	file_dir = gr.Textbox(placeholder = "file_dir")
	file_obj = gr.File()
	file_sep = gr.Radio(["csv", "tsv"], label = "CSV File Sep type", info="CSV File sep type")

	src_col_name = gr.Textbox("source", placeholder = "src column name")
	label_col_name = gr.Textbox("label", placeholder = "label column name")

	## Perf Configs
	num_labels = gr.Textbox(2, placeholder = "Number of labels")
	target_label = gr.Textbox(1, placeholder = "target_label")

	num_confidence_bins = gr.Textbox(20, placeholder = "num_confidence_bins")

	##
	inputs = [
		## Model
		model_dir,
		device_type,
		batch_size,
		## Files
		file_dir,
		file_obj,
		file_sep,
		src_col_name,
		label_col_name,
		num_labels,
		target_label,
		num_confidence_bins
	]
	interface = gr.Interface(fn=predict_with_eval_file, inputs=inputs, outputs=["json", "plot", "dataframe"])
	return interface

if __name__ == "__main__":
	interface = make_clf_perf_eval_interface()
	interface.launch()