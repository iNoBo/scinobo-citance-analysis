"""

SCINOBO CITANCE ANALYSIS BULK INFERENCE SCRIPT

"""

import os
import re
import json
import fnmatch
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

m_name = 'Flan-T5-Base-Citance_Analysis_Lora'

from citance_analysis.pipeline.model import classify_text, classify_text_batch

print('Loaded Model!')

grobid_url = os.getenv('GROBID_URL')
if not grobid_url:
    grobid_url = 'https://kermitt2-grobid.hf.space/api/processFulltextDocument'
else:
    grobid_url = f'{grobid_url}/api/processFulltextDocument'

cit_stopwords = ('leg', 'legend', 'tab', 'tabs', 'table', 'tables', 'fig', 'figs', 'figure', 'figures', 'note', 'notes', 'sup', 'suppl', 'supplementary', 'app', 'appx', 'append', 'appendix', 'ver', 'version', 'footnote', 'see', 'sec', 'section', 'cohort', 'ext', 'extended', 'additional')


def parse_pdf(pdf_path):
    """
    > node names in the TEI XML file:
        ['TEI', 'abstract', 'affiliation', 'analytic', 'appInfo', 'application', 'author', 'availability', 'back', 'biblScope', 'biblStruct', 
        'body', 'cell', 'date', 'desc', 'div', 'email', 'encodingDesc', 'facsimile', 'figDesc', 'figure', 'fileDesc', 'forename', 'formula', 
        'graphic', 'head', 'idno', 'imprint', 'label', 'licence', 'listBibl', 'monogr', 'note', 'orgName', 'p', 'persName', 'profileDesc', 
        'pubPlace', 'publicationStmt', 'publisher', 'ref', 'respStmt', 'row', 's', 'sourceDesc', 'surface', 'surname', 'table', 'teiHeader', 
        'text', 'title', 'titleStmt']
    
    """
    with open(pdf_path, 'rb') as fin:
        pdf_data = fin.read()
    
    payload = {
        'input': pdf_data,
        'consolidateHeader': 1,
        'consolidateCitations': 1,
        'includeRawCitations': 1,
        'includeRawAffiliations': 1,
        'teiCoordinates': 1,
        'segmentSentences': 1
    }

    res = requests.post(grobid_url, files=payload)
    
    pdf_metadata = parse_tei_xml(res.text, is_file=False)

    return pdf_metadata


def parse_tei_xml(tei_xml, is_file=True):
    """
    > node names in the TEI XML file:
        ['TEI', 'abstract', 'affiliation', 'analytic', 'appInfo', 'application', 'author', 'availability', 'back', 'biblScope', 'biblStruct', 
        'body', 'cell', 'date', 'desc', 'div', 'email', 'encodingDesc', 'facsimile', 'figDesc', 'figure', 'fileDesc', 'forename', 'formula', 
        'graphic', 'head', 'idno', 'imprint', 'label', 'licence', 'listBibl', 'monogr', 'note', 'orgName', 'p', 'persName', 'profileDesc', 
        'pubPlace', 'publicationStmt', 'publisher', 'ref', 'respStmt', 'row', 's', 'sourceDesc', 'surface', 'surname', 'table', 'teiHeader', 
        'text', 'title', 'titleStmt']
    
    """

    if is_file:
        # Load the TEI XML file
        with open(tei_xml, 'r', encoding='utf-8') as fin:
            tei_xml_data = fin.read()
    else:
        # The function has TEI XML data directly
        tei_xml_data = tei_xml

    soup = BeautifulSoup(tei_xml_data, 'xml')

    pdf_metadata = dict()

    if soup.find('sourceDesc'):
        identifiers = [[x.attrs['type'], x.text] for x in soup.find('sourceDesc').find_all('idno')]
    else:
        identifiers = None
    pdf_metadata['identifiers'] = identifiers if identifiers else None

    title = soup.find('title')
    pdf_metadata['title'] = title.text if title else None

    abstract = soup.find('abstract')
    abstract = [[s.text for s in p.find_all('s')] for p in abstract.find_all('p')]  # find all paragraphs and then all sentences
    pdf_metadata['abstract'] = abstract if abstract else None

    sections = soup.find_all('div')
    # For each section, find the head (if there is one), all paragraphs and then all sentences
    sections = [[s.head.text if s.head else None, [[[s, s.text] for s in p.find_all('s')] for p in s.find_all('p')]] for s in sections]

    # When a section has no head, then search if it is a figure, table or graphic using the "find_parent" method
    for i, section in enumerate(sections):
        if (section[0] is None) and (len(section[1])!=0) and (section[1][0] != []):
            sec_first_s = section[1][0][0][0]
            if sec_first_s.find_parent('figure'):
                par = sec_first_s.find_parent('figure')
                sections[i][0] = par.find('head').text if par.find('head') else 'figure'
            elif sec_first_s.find_parent('table'):
                par = sec_first_s.find_parent('table')
                sections[i][0] = par.find('head').text if par.find('head') else 'table'
            elif sec_first_s.find_parent('graphic'):
                par = sec_first_s.find_parent('graphic')
                sections[i][0] = par.find('head').text if par.find('head') else 'graphic'

    # Remove the extra from the sentences
    for i, section in enumerate(sections):
        for j, paragraph in enumerate(section[1]):
            for k, sentence in enumerate(paragraph):
                sections[i][1][j][k] = sentence[1]
    
    pdf_metadata['sections'] = sections

    # Find the references in the bibliography
    bibl_references = [{
        'id': r.attrs['xml:id'] if 'xml:id' in r.attrs else None,
        'doi': r.find('idno', {'type': 'DOI'}).text.lower() if r.find('idno', {'type': 'DOI'}) else None,
        'title': r.find('analytic').find('title').text if r.find('analytic') and r.find('analytic').find('title') else None,
        'authors': [{
            'surname': a.find('surname').text if a.find('surname') else None,
            'forename': a.find('forename').text if a.find('forename') else None,
            # 'email': a.find('email').text if a.find('email') else None,
            # 'affiliation': {
            #     'id': a.find('affiliation').attrs['key'] if 'key' in a.find('affiliation').attrs else None,
            #     'name': a.find('orgName').text if a.find('orgName') else None,
            # } if a.find('affiliation') else None
            } for a in r.find('analytic').find_all('author')] if r.find('analytic') else None,
        } for r in soup.find_all('biblStruct')]

    # Find the figures in the text
    figures = [{
        'id': x.attrs['xml:id'] if 'xml:id' in x.attrs else None, 
        'head': x.find('head').text if x.find('head') else None,
        'desc': x.find('figDesc').text if x.find('figDesc') else None
        } for x in soup.find_all('figure')]

    # Find the formulas in the text
    formulas = [{
        'id': x.attrs['xml:id'] if 'xml:id' in x.attrs else None,
        'desc': x.text,
        } for x in soup.find_all('formula')]


    # Find the citances in the section, paragraph and sentence
    citances = [[[[sec_i, par_i, s_i, s.find_all('ref')] for s_i, s in enumerate(par.find_all('s')) if s.find('ref')] for par_i, par in enumerate(sec.find_all('p')) if par.find('ref')] for sec_i, sec in enumerate(soup.find_all('div')) if sec.find('ref')]
    citances = [i2 for s2 in [i for s in citances for i in s] for i2 in s2]
    citances = [{
            'sec_idx': x[0], 
            'par_idx': x[1], 
            's_idx': x[2], 
            'refs':
                [{
                'target': y.attrs['target'] if 'target' in y.attrs else None,
                'type': y.attrs['type'] if 'type' in y.attrs else None,
                'text': y.text,} for y in x[3]
                ],
            'sentence': sections[x[0]][1][x[1]][x[2]]} for x in citances]

    pdf_metadata['bibl_references'] = bibl_references
    pdf_metadata['figures'] = figures
    pdf_metadata['formulas'] = formulas
    pdf_metadata['citances'] = citances

    return pdf_metadata


def find_polarity(citance, citation_mark):
    prompt = f"### Snipet: {citance} ### Citation Mark: {citation_mark} ### Question: What is the polarity of the citation? ### Answer:"
    choices = ['Supporting', 'Neutral', 'Refuting']
    json_data = {
        'text': prompt,
        'choices': choices,
        'top_k': 50,
        'gen_config': {
            'max_new_tokens': 256
    }
    }
    return classify_text(json_data['text'], json_data['choices'], json_data['top_k'], **json_data['gen_config'])


def find_intent(citance, citation_mark):
    prompt = f"### Snipet: {citance} ### Citation Mark: {citation_mark} ### Question: What is the intent of the citation? ### Answer:"
    choices = ['Comparison', 'Reuse', 'Generic']
    json_data = {
        'text': prompt,
        'choices': choices,
        'top_k': 500,
        'gen_config': {
            'max_new_tokens': 256
    }
    }
    return classify_text(json_data['text'], json_data['choices'], json_data['top_k'], **json_data['gen_config'])


def find_semantics(citance, citation_mark):
    prompt = f"### Snipet: {citance} ### Citation Mark: {citation_mark} ### Question: What are the semantics of the citation? ### Answer:"
    choices = ['Artifact', 'Claim', 'Results', 'Methodology']
    json_data = {
        'text': prompt,
        'choices': choices,
        'top_k': 500,
        'gen_config': {
            'max_new_tokens': 256
    }
    }
    return classify_text(json_data['text'], json_data['choices'], json_data['top_k'], **json_data['gen_config'])


def find_polarity_batch(citances):
    choices = ['Supporting', 'Neutral', 'Refuting']
    prompts = []
    for citance, citation_mark in citances:
        prompt = f"### Snipet: {citance} ### Citation Mark: {citation_mark} ### Question: What is the polarity of the citation? ### Answer:"
        prompts.append(prompt)
    json_data = {
        'gen_config': {
            'max_new_tokens': 256
        }
    }
    return classify_text_batch(prompts=prompts, choices=choices, top_k=256, **json_data['gen_config'])


def find_intent_batch(citances):
    choices = ['Comparison', 'Reuse', 'Generic']
    prompts = []
    for citance, citation_mark in citances:
        prompt = f"### Snipet: {citance} ### Citation Mark: {citation_mark} ### Question: What is the intent of the citation? ### Answer:"
        prompts.append(prompt)
    json_data = {
        'gen_config': {
            'max_new_tokens': 256
        }
    }
    return classify_text_batch(prompts=prompts, choices=choices, top_k=256, **json_data['gen_config'])


def find_semantics_batch(citances):
    choices = ['Artifact', 'Claim', 'Results', 'Methodology']
    prompts = []
    for citance, citation_mark in citances:
        prompt = f"### Snipet: {citance} ### Citation Mark: {citation_mark} ### Question: What are the semantics of the citation? ### Answer:"
        prompts.append(prompt)
    json_data = {
        'gen_config': {
            'max_new_tokens': 256
        }
    }
    return classify_text_batch(prompts=prompts, choices=choices, top_k=256, **json_data['gen_config'])


def find_citations_old(text, stopwords=cit_stopwords):
    # TODO: some of the matches are duplicates
    matches_0 = re.findall(r'[[(][^\]\[)(]*?et al.*?(?:\]|\)|$)', text)
    matches_1 = re.findall(r'(\[\s*\d+\s*(?!\s*(?:\w|[^\w,\-\]])|\s*,\d+(?:[^\w,\-\]]))\s*?(?:\]|$))', text)
    matches_1b = re.findall(r'(\[\s*[A-Z](?:\w+\s+(?:\&\s+)?){1,}\d+\s*(?:\]|$))', text)
    matches_2 = re.findall(r'(\(\s*\d+\s*(?!\s*(?:\w|[^\w,\-\)])|\s*,\d+(?:[^\w,\-\)]))\s*?(?:\)|$))', text)
    matches_2b = re.findall(r'(\(\s*[A-Z](?:\w+\s+(?:\&\s+)?){1,}\d+\s*(?:\)|$))', text)
    # 1b, 2b are noisy, remove some stopwords
    matches_1b = [e for e in matches_1b if not any([e2.lower().strip() in stopwords for e2 in e.split()])]
    matches_2b = [e for e in matches_2b if not any([e2.lower().strip() in stopwords for e2 in e.split()])]
    return matches_0, matches_1, matches_1b, matches_2, matches_2b


def find_citations(text, stopwords=cit_stopwords):
    # TODO: some of the matches are duplicates
    matches_0 = re.findall(r'[[(][^\]\[)(]*?et al.*?(?:\]|\)|$)', text)
    matches_1 = re.findall(r'(\[\s*\d+\s*(?!\s*(?:\w|[^\w,\-\]])|\s*,\d+(?:[^\w,\-\]]))\s*?(?:\]|$))', text)
    matches_1b = re.findall(r'(\[\s*[A-Z](?:\w+\s+(?:\&\s+)?){1,}\d+\s*(?:\]|$))', text)
    matches_1c = re.findall(r'(\[\s*(?:(?:\d+(?:-\d+)?)(?:\s*,\s*(?:\d+(?:-\d+)?))*)\s*\])', text)
    matches_2 = re.findall(r'(\(\s*\d+\s*(?!\s*(?:\w|[^\w,\-\)])|\s*,\d+(?:[^\w,\-\)]))\s*?(?:\)|$))', text)
    matches_2b = re.findall(r'(\(\s*[A-Z](?:\w+\s+(?:\&\s+)?){1,}\d+\s*(?:\)|$))', text)
    matches_2c = re.findall(r'(\(\s*(?:(?:\d+(?:-\d+)?)(?:\s*,\s*(?:\d+(?:-\d+)?))*)\s*\))', text)

    # Matches [1 and 1c] , [2 and 2c] can be combined
    matches_1 = list(set(matches_1 + matches_1c))
    matches_2 = list(set(matches_2 + matches_2c))

    # 1b, 2b are noisy, remove some stopwords
    matches_1b = [e for e in matches_1b if not any([e2.lower().strip() in stopwords for e2 in e.split()])]
    matches_2b = [e for e in matches_2b if not any([e2.lower().strip() in stopwords for e2 in e.split()])]
    return matches_0, matches_1, matches_1b, matches_2, matches_2b


def find_pis(citance, citation_mark):
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
    return results


def find_pis_batch(citances):
    all_results = []
    all_polarity = find_polarity_batch(citances)
    all_intent = find_intent_batch(citances)
    all_semantics = find_semantics_batch(citances)
    for i in range(len(citances)):
        polarity = all_polarity[i]
        intent = all_intent[i]
        semantics = all_semantics[i]
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
        all_results.append(results)
    return all_results


def find_mark_pis(citance):
    matches_0, matches_1, matches_1b, matches_2, matches_2b = find_citations(citance)
    citation_marks = sorted(set(matches_0 + matches_1 + matches_1b + matches_2 + matches_2b))
    results = {}
    citances = []
    for citation_mark in citation_marks:
        citances.append((citance, citation_mark))
    pis_results = find_pis_batch(citances)
    for i, citation_mark in enumerate(citation_marks):
        results[citation_mark] = pis_results[i]
    return results


def get_citation_marks(citance):
    matches_0, matches_1, matches_1b, matches_2, matches_2b = find_citations(citance)
    citation_marks = sorted(set(matches_0 + matches_1 + matches_1b + matches_2 + matches_2b))
    return citation_marks


def extract_citances_pis(pdf_file, xml_mode=False):
    """Extract the polarity and intent from the citances in the publication using batch processing."""

    # Get the metadata
    if xml_mode:
        print(f"Processing TEI XML: {pdf_file}")
        pdf_metadata = parse_tei_xml(pdf_file)
    else:
        print(f"Processing PDF: {pdf_file}")
        pdf_metadata = parse_pdf(pdf_file)

    # Initialize the variables
    citance_pairs = []
    citances_map = []

    # Get the citances
    citances = pdf_metadata['citances']
    for citance_idx, citance in enumerate(citances):
        for ref_idx, ref in enumerate(citance['refs']):
            if ref['target'] and '#b' in ref['target']:
                # Collect citance sentence and reference text pairs for batch processing
                citance_pairs.append((citance['sentence'], ref['text']))
                # Keep a map to where to assign the results later
                citances_map.append((citance_idx, ref_idx))

    # Print the number of citances to process
    print(f"Number of citances to process: {len(citance_pairs)}")
    
    # Process the pairs in batch
    batch_results = find_pis_batch(citance_pairs)

    # Initialize the citances results
    res_citances = []

    # Assign the results back and prepare the final structure
    for i, (citance_idx, ref_idx) in enumerate(citances_map):
        citance = citances[citance_idx]
        ref = citance['refs'][ref_idx]
        results = batch_results[i]

        ref['results'] = results
        res_citances.append({
            'citance': citance['sentence'],
            'citation_mark': ref['text'],
            'results': results
        })
    
    return {
        'pdf_metadata': pdf_metadata,
        'res_citances': res_citances
    }


def find_mark_pis_parquet(citance):
    matches_0, matches_1, matches_1b, matches_2, matches_2b = find_citations(citance)
    citation_marks = sorted(set(matches_0 + matches_1 + matches_1b + matches_2 + matches_2b))
    # Add an empty citation mark
    citation_marks.append('')
    results = {}
    citances = []
    for citation_mark in citation_marks:
        citances.append((citance, citation_mark))
    pis_results = find_pis_batch(citances)
    for i, citation_mark in enumerate(citation_marks):
        results[citation_mark] = pis_results[i]
    return results


def infer_parquet(input_dir, output_dir, filter_input=None):
    print("Input DIR with Parquet files...")
    print(input_dir)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter the parquet files
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

    if filter_input:
        parquet_files = fnmatch.filter(parquet_files, filter_input)

    # Read the parquet files
    for parquet_file in tqdm(parquet_files):
        parquet_path = os.path.join(input_dir, parquet_file)
        parquet_df = pd.read_parquet(parquet_path)

        # Process each row
        all_outputs = []
        for idx, row in tqdm(parquet_df.iterrows(), total=parquet_df.shape[0]):
            cit_id = row['id']
            cit_citances = row['citation_mentions']
            results = []
            for citance in cit_citances:
                citance_results = find_mark_pis_parquet(citance)
                results.append(citance_results)
            
            all_outputs.append({
                'id': cit_id,
                'citation_mentions': cit_citances.tolist(),
                'results': results
            })
        
        with open(os.path.join(output_dir, parquet_file.replace('.parquet', '.json')), 'w', encoding='utf-8') as fout:
                json.dump(all_outputs, fout, indent=1)

        print("Parquet processed and results saved in the output directory...")
        print(output_dir)
        print("Done!")


def infer_pdf(input_dir, output_dir, xml_mode=False, filter_input=None):
    if xml_mode:
        print("Input DIR with TEI XML files...")
    else:
        print("Input DIR with PDFs...")
    print(input_dir)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if xml_mode:
        # Get the list of XML files
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    else:
        # Get the list of PDF files
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    if filter_input:
        pdf_files = fnmatch.filter(pdf_files, filter_input)

    # Process each PDF file
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(input_dir, pdf_file)
        pdf_metadata = extract_citances_pis(pdf_path, xml_mode)

        # Save the results
        if xml_mode:
            with open(os.path.join(output_dir, pdf_file.replace('.xml', '.json')), 'w', encoding='utf-8') as fout:
                json.dump(pdf_metadata, fout, indent=1)
        else:
            with open(os.path.join(output_dir, pdf_file.replace('.pdf', '.json')), 'w', encoding='utf-8') as fout:
                json.dump(pdf_metadata, fout, indent=1)

    if xml_mode:
        print("TEI XMLs processed and results saved in the output directory...")
    else:
        print("PDFs processed and results saved in the output directory...")
    print(output_dir)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Bulk inference of citances polarity and intent from PDF files.')
    parser.add_argument('--input_dir', type=str, help='Directory with PDF/XML files to process.', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory to save the results.', required=True)
    parser.add_argument('--xml_mode', action='store_true', help='Process TEI XML files instead of PDF files.')
    parser.add_argument('--parquet_mode', action='store_true', help='Run the pipeline for parquet files instead of PDFs that contain the columns: "id", "citation_mentions" .')
    parser.add_argument('--filter_input', type=str, help='Wildcard pattern to filter input files to analyze.')
    args = parser.parse_args()

    if args.parquet_mode:
        infer_parquet(args.input_dir, args.output_dir, args.filter_input)
    else:
        infer_pdf(args.input_dir, args.output_dir, args.xml_mode, args.filter_input)


if __name__ == '__main__':
    main()    
