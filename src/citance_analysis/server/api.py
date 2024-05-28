"""

SCINOBO CITANCE ANALYSIS API SERVER

"""

import os
import traceback
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
from fastapi.middleware.cors import CORSMiddleware

m_name = 'Flan-T5-Base-Citance_Analysis_Lora'

from citance_analysis.pipeline.inference import find_mark_pis, find_polarity, find_intent, find_semantics, find_pis_batch, get_citation_marks
from citance_analysis.server.logging_setup import setup_root_logger

print('Loaded Model!')

app = FastAPI()

# init the logger
setup_root_logger()
logger = logging.getLogger(__name__)
logger.info("Citance Analysis Api initialized")

class CitanceInferRequest(BaseModel):
    citance: str
    citation_mark: str = Field('')

class CitanceInferResponse(BaseModel):
    citance: str
    output: List[Dict[str, Any]]

class CitanceListInferRequest(BaseModel):
    citances: List[CitanceInferRequest]

class CitanceListInferResponse(BaseModel):
    output: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    success: int
    message: str

app = FastAPI()

# handle CORS -- at a later stage we can restrict the origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/infer_citance', response_model=CitanceInferResponse, responses={400: {"model": ErrorResponse}})
def infer_citance(request_data: CitanceInferRequest):
    try:
        logger.debug("JSON received...")
        logger.debug(request_data.json())

        citance = request_data.citance
        citation_mark = request_data.citation_mark

        if not citation_mark:
            # List of results for each citation mark
            results = []
            marks = find_mark_pis(citance)
            for mark, pis in marks.items():
                results.append({
                    'citation_mark': mark,
                    'results': pis
                })

            return CitanceInferResponse(citance=citance, output=results)

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

        return CitanceInferResponse(citance=citance, output=[{'citation_mark': citation_mark, 'results': results}])

    except Exception as e:
        logger.error(str(e))
        return HTTPException(status_code=400, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})


@app.post('/infer_citances', response_model=CitanceListInferResponse, responses={400: {"model": ErrorResponse}})
def infer_citances(request_data: CitanceListInferRequest):
    try:
        logger.debug("JSON received...")
        logger.debug(request_data.json())

        citances = request_data.citances
        all_citance_pairs = []
        all_results = []
        citance_indices = []
        for _i, _citance in enumerate(citances):
            if not _citance.citation_mark:
                citation_marks = get_citation_marks(_citance.citance)
            else:
                citation_marks = [_citance.citation_mark]
            for citation_mark in citation_marks:
                all_citance_pairs.append((_citance.citance, citation_mark))
                citance_indices.append(_i)
            
        all_results = find_pis_batch(all_citance_pairs)

        # Aggregate the results per citance using the citance_indices to match the CitanceListInferResponse
        all_results_dict = []
        prev_citance_index = -1
        for i, ar in enumerate(all_results):
            citance_index = citance_indices[i]
            if citance_index != prev_citance_index:
                all_results_dict.append({
                    'citance': citances[citance_index].citance,
                    'output': []
                })
                prev_citance_index = citance_index
            all_results_dict[-1]['output'].append({
                'citation_mark': all_citance_pairs[i][1],
                'results': ar
            })

        return CitanceListInferResponse(output=all_results_dict)

    except Exception as e:
        logger.error(str(e))
        return HTTPException(status_code=400, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('PORT', 8000))
