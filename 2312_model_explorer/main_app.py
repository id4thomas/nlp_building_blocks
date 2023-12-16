import gradio as gr
from seq_clf.seq_calibration import make_calibration_eval_interface

# i1 = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
# i2 = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
# demo = gr.TabbedInterface([i1, i2], ["Text-to-speech", "Speech-to-text"])
i1 = make_calibration_eval_interface()
demo = gr.TabbedInterface([i1], ["Sequence CLF Calibration"])
if __name__ == "__main__":
    demo.launch()
