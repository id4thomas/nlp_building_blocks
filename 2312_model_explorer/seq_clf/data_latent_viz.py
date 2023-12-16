import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
	from utils.inference_utils import load_model_and_tokenizer, extract_hid
	from utils.latent_utils import reduce_dim_with_tsne
except ModuleNotFoundError as e:
	from seq_clf.utils.inference_utils import load_model_and_tokenizer, extract_hid
	from seq_clf.utils.latent_utils import reduce_dim_with_tsne
except Exception as e:
	raise e

def latent_viz_with_file(
		model_dir, 
		device,
		batch_size,
		file_dir,
		file_obj,
		file_sep,
		data_sample_size,
		src_col_name,
		label_col_name,
		# num_labels,
		progress = gr.Progress()
	):
	progress(0, desc="Starting...")

	## Load CLF Model
	# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	model, tokenizer = load_model_and_tokenizer(model_dir, device = device, model_type = "plm")
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
	df = df.sample(min(df.shape[0], data_sample_size))
	print("Sampled DF",df.shape, df.columns)

	sources = df[src_col_name].values.tolist()
	y_true = np.array(df[label_col_name].values.tolist())
	predicted = extract_hid(
		model,
		tokenizer,
		sources,
		device = device,
		batch_size = batch_size,
		seq_max_length = 512
	).numpy()

	X_embedded = reduce_dim_with_tsne(predicted)
	X_embedded = np.array(X_embedded)

	fig, ax = plt.subplots()
	ax.set_title("T-SNE dim reduc")
	unique_labels = df[label_col_name].unique()
	for l in unique_labels:
		X_embedded_l = X_embedded[y_true==l, :]
		ax.scatter(X_embedded_l[:,0], X_embedded_l[:, 1], alpha=0.5, label = l)
	ax.set_xlabel('Dimension 1')
	ax.set_ylabel('Dimension 2')
	ax.legend()

	progress(1.0, desc="Complete")
	return fig

def make_latent_viz_interface():
	model_dir = gr.Textbox(placeholder = "model_dir")
	device_type = gr.Radio(["cpu", "cuda", "mps"], label="Device Type", info="Accelerator device type")
	batch_size = gr.Slider(8, 1024, value=16, label="Batch Size", info="between 8 ~ 1024")

	file_dir = gr.Textbox(placeholder = "file_dir")
	file_obj = gr.File()
	file_sep = gr.Radio(["csv", "tsv"], label = "CSV File Sep type", info="CSV File sep type")
	data_sample_size = gr.Slider(2000, 20000, value=2000, label="Data Sample Size", info="between 2k ~ 20k")

	src_col_name = gr.Textbox("source", placeholder = "src column name")
	label_col_name = gr.Textbox("label", placeholder = "label column name")

	## Perf Configs
	# num_labels = gr.Textbox(2, placeholder = "Number of labels")

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
		data_sample_size,
		src_col_name,
		label_col_name,
		# num_labels
	]
	interface = gr.Interface(fn=latent_viz_with_file, inputs=inputs, outputs=["plot"])
	return interface

if __name__ == "__main__":
	interface = make_latent_viz_interface()
	interface.launch()