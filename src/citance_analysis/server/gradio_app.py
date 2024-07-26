
import os
import gradio as gr
import requests as req
from tqdm import tqdm
from citance_analysis.pipeline.inference import extract_citances_pis, find_mark_pis, find_mark_pis_parquet, find_polarity, find_intent, find_semantics

# Retrieve HF space secrets
BACKEND_IP = os.getenv('BACKEND_IP')
BACKEND_PORT = os.getenv('BACKEND_PORT')
BACKEND_PATH = os.getenv('BACKEND_PATH')

# Define the functions to handle the inputs and outputs
def analyze_text(citance, citation_mark, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        if not citation_mark:
            # List of results for each citation mark
            results = []
            marks = find_mark_pis(citance)
            for i, (mark, pis) in enumerate(marks.items()):
                results.append({
                    'citation_mark': mark,
                    'results': pis
                })
        else:
            polarity = find_polarity(citance, citation_mark)
            intent = find_intent(citance, citation_mark)
            semantics = find_semantics(citance, citation_mark)
            results = {
                'polarity': max(polarity, key=polarity.get),
                'intent': max(intent, key=intent.get),
                'semantics': max(semantics, key=semantics.get),
                'scores': {
                    'polarity': polarity,
                    'intent': intent,
                    'semantics': semantics
                }
            }
    except Exception as e:
        results = {'error': str(e)}
    return results

def analyze_pdf(pdf_file, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        results = extract_citances_pis(pdf_file, xml_mode=False)['res_citances']
    except Exception as e:
        results = {'error': str(e)}
    return results


def analyze_input_doi(doi: str | None, progress=gr.Progress(track_tqdm=True)):
    if (doi is None):
        results = {'error': 'Please provide the DOI of the publication'}
        return results
    if (doi == ''):
        results = {'error': 'Please provide the DOI of the publication'}
        return results
    try:
        url = f"http://{BACKEND_IP}:{BACKEND_PORT}{BACKEND_PATH}{doi}"
        response = req.get(url)
        response.raise_for_status()
        # Get the data
        data = response.json()

        # Extract the citances and PIs
        results = []
        for citance in tqdm(data['context'], desc='Processing Citances'):
            res = find_mark_pis_parquet(citance)
            results.append({
                'citance': citance,
                'results': res
            })

        return data, results
    except Exception as e:
        results = {'error': str(e)}
        return results, results

# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### SciNoBo Citance Analysis - Text Mode")
    citance_input = gr.Textbox(label="Citance")
    citation_mark_input = gr.Textbox(label="Citation Mark", placeholder="Enter citation mark or leave blank")
    process_text_button = gr.Button("Process")

    text_output = gr.JSON(label="Output")

    process_text_button.click(analyze_text, inputs=[citance_input, citation_mark_input], outputs=[text_output])

# Define the interface for the second tab (PDF Analysis)
with gr.Blocks() as pdf_analysis:
    gr.Markdown("### SciNoBo Citance Analysis - PDF Mode")
    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_pdf_button = gr.Button("Process")

    pdf_output = gr.JSON(label="Output")

    process_pdf_button.click(analyze_pdf, inputs=[pdf_input], outputs=[pdf_output])

# Define the interface for the third tab (DOI Mode)
with gr.Blocks() as doi_mode:
    gr.Markdown("### SciNoBo Citance Analysis - DOI Mode")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier")
    process_doi_button = gr.Button("Process")

    doi_metadata = gr.JSON(label="DOI Metadata")
    doi_output = gr.JSON(label="Output")
    
    process_doi_button.click(analyze_input_doi, inputs=[doi_input], outputs=[doi_metadata, doi_output])

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis, pdf_analysis, doi_mode], ["Text Mode", "PDF Mode", "DOI Mode"])

# Launch the interface
demo.queue().launch()
