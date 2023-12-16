import gradio as gr

## Import interfances
from clf_perf import make_clf_perf_eval_interface
from data_latent_viz import make_latent_viz_interface

if __name__ == "__main__":
	clf_perf_interface = make_clf_perf_eval_interface()
	latent_viz_interface = make_latent_viz_interface()

	demo = gr.TabbedInterface(
		[clf_perf_interface, latent_viz_interface], 
		["Clf Performance", "Latent Viz"]
	)
	demo.launch()