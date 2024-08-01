
import os
import gradio as gr
import pandas as pd
from tqdm import tqdm
import requests as req
from citance_analysis.pipeline.inference import extract_citances_pis, find_mark_pis, find_mark_pis_parquet, find_polarity, find_intent, find_semantics

# Retrieve HF space secrets
BACKEND_IP = os.getenv('BACKEND_IP')
BACKEND_PORT = os.getenv('BACKEND_PORT')
BACKEND_PATH = os.getenv('BACKEND_PATH')


def convert_results_to_table(citance, results):
    table = []
    if isinstance(results, list):
        for res in results:
            if 'results' in res:
                table.append({
                    'Citance': citance,
                    'Citation Mark': res['citation_mark'],
                    'Polarity': res['results']['polarity'],
                    'Intent': res['results']['intent'],
                    'Semantics': res['results']['semantics'],
                    'Scores for Polarity': str(res['results']['scores']['polarity']),
                    'Scores for Intent': str(res['results']['scores']['intent']),
                    'Scores for Semantics': str(res['results']['scores']['semantics'])
                })
            else:
                table.append({
                    'Citance': citance,
                    'Citation Mark': res['citation_mark'],
                    'Polarity': res['polarity'],
                    'Intent': res['intent'],
                    'Semantics': res['semantics'],
                    'Scores for Polarity': str(res['scores']['polarity']),
                    'Scores for Intent': str(res['scores']['intent']),
                    'Scores for Semantics': str(res['scores']['semantics'])
                })
    else:
        if 'results' in results:
            table.append({
                'Citance': citance,
                'Citation Mark': results['citation_mark'],
                'Polarity': results['results']['polarity'],
                'Intent': results['results']['intent'],
                'Semantics': results['results']['semantics'],
                'Scores for Polarity': str(results['results']['scores']['polarity']),
                'Scores for Intent': str(results['results']['scores']['intent']),
                'Scores for Semantics': str(results['results']['scores']['semantics'])
            })
        else:
            table.append({
                'Citance': citance,
                'Citation Mark': results['citation_mark'],
                'Polarity': results['polarity'],
                'Intent': results['intent'],
                'Semantics': results['semantics'],
                'Scores for Polarity': str(results['scores']['polarity']),
                'Scores for Intent': str(results['scores']['intent']),
                'Scores for Semantics': str(results['scores']['semantics'])
            })

    df = pd.DataFrame(table)
    return df


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
                'citation_mark': citation_mark,
                'polarity': max(polarity, key=polarity.get),
                'intent': max(intent, key=intent.get),
                'semantics': max(semantics, key=semantics.get),
                'scores': {
                    'polarity': polarity,
                    'intent': intent,
                    'semantics': semantics
                }
            }
        results = convert_results_to_table(citance, results)
    except Exception as e:
        results = {'error': str(e)}
    return results

def analyze_pdf(pdf_file, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        results = extract_citances_pis(pdf_file, xml_mode=False)['res_citances']
        res_dataframes = [convert_results_to_table(x['citance'], x) for x in results]

        # Combine the results
        res_dataframe = pd.concat(res_dataframes)

        return res_dataframe
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
        
        # Bring the results into the same format
        res_dataframes = []
        for res in results:
            res_citance = res['citance']
            res_results = []
            for res_result in res['results']:
                res_results.append({
                    'citation_mark': res_result,
                    'polarity': res['results'][res_result]['polarity'],
                    'intent': res['results'][res_result]['intent'],
                    'semantics': res['results'][res_result]['semantics'],
                    'scores': {
                        'polarity': res['results'][res_result]['scores']['polarity'],
                        'intent': res['results'][res_result]['scores']['intent'],
                        'semantics': res['results'][res_result]['scores']['semantics']
                    }
                })
            res_results = convert_results_to_table(res_citance, res_results)
            res_dataframes.append(res_results)
        
        # Combine the results
        res_dataframe = pd.concat(res_dataframes)

        return data, res_dataframe
    except Exception as e:
        results = {'error': str(e)}
        return results, results

# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### SciNoBo Citance Analysis - Text Mode")
    citance_input = gr.Textbox(label="Citance")
    citation_mark_input = gr.Textbox(label="Citation Mark", placeholder="Enter citation mark or leave blank")
    process_text_button = gr.Button("Process")

    text_output = gr.DataFrame(label="Output", headers=['Citance', 'Citation Mark', 'Polarity', 'Intent', 'Semantics'], row_count=1)

    process_text_button.click(analyze_text, inputs=[citance_input, citation_mark_input], outputs=[text_output])

# Define the interface for the second tab (PDF Analysis)
with gr.Blocks() as pdf_analysis:
    gr.Markdown("### SciNoBo Citance Analysis - PDF Mode")
    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_pdf_button = gr.Button("Process")

    pdf_output = gr.DataFrame(label="Output", headers=['Citance', 'Citation Mark', 'Polarity', 'Intent', 'Semantics'], row_count=1)

    process_pdf_button.click(analyze_pdf, inputs=[pdf_input], outputs=[pdf_output])

# Define the interface for the third tab (DOI Mode)
with gr.Blocks() as doi_mode:
    gr.Markdown("### SciNoBo Citance Analysis - DOI Mode")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier")
    process_doi_button = gr.Button("Process")

    doi_metadata = gr.JSON(label="DOI Metadata")

    doi_output = gr.DataFrame(label="Output", headers=['Citance', 'Citation Mark', 'Polarity', 'Intent', 'Semantics'], row_count=1)

    process_doi_button.click(analyze_input_doi, inputs=[doi_input], outputs=[doi_metadata, doi_output])

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis, pdf_analysis, doi_mode], ["Text Mode", "PDF Mode", "DOI Mode"])

# Launch the interface
demo.queue().launch()
