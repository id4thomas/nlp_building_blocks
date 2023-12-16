import gradio as gr
from clf_perf import make_clf_perf_eval_interface

if __name__ == "__main__":
	clf_perf_interface = make_clf_perf_eval_interface()
	demo = gr.TabbedInterface(
		[clf_perf_interface], 
		["Clf Performance"]
	)
	demo.launch()